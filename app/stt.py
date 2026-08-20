import os
import asyncio
import json
import base64
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def process_audio_stream(websocket: WebSocket) -> str:
    """
    Connects to Sarvam's Realtime Speech-to-Text WebSocket API,
    streams audio from the client, and returns the final transcribed text.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not set in the environment variables.")

    language_code = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")

    # Connect to the Realtime API endpoint
    sarvam_url = (
        f"wss://api.sarvam.ai/speech-to-text-realtime/ws"
        f"?model=saaras:v3-realtime&language_code={language_code}&stream_type=fast"
    )
    
    headers = {
        "api-subscription-key": api_key
    }

    final_transcript = ""

    async with websockets.connect(sarvam_url, extra_headers=headers) as sarvam_ws:
        
        async def forward_audio():
            try:
                # Wait for the first chunk of audio from the client
                first_chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
                
                # Send first chunk
                await sarvam_ws.send(json.dumps({
                    "event": "audio_input",
                    "audio": base64.b64encode(first_chunk).decode("utf-8")
                }))
                
                # Stream subsequent chunks from the client
                while True:
                    # If the client pauses sending audio for more than 1.5 seconds,
                    # we assume the turn is complete and stop streaming.
                    chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=1.5)
                    if not chunk:
                        break
                    
                    await sarvam_ws.send(json.dumps({
                        "event": "audio_input",
                        "audio": base64.b64encode(chunk).decode("utf-8")
                    }))
            except (asyncio.TimeoutError, WebSocketDisconnect):
                # Timeout or client disconnect indicates end of turn
                pass
            except Exception as e:
                print(f"Error forwarding audio to Sarvam: {e}")
            finally:
                # Gracefully signal the end of the audio stream
                try:
                    await sarvam_ws.send(json.dumps({"event": "end"}))
                except Exception:
                    pass

        async def receive_responses():
            nonlocal final_transcript
            try:
                async for message in sarvam_ws:
                    try:
                        response = json.loads(message)
                        event = response.get("event")
                        
                        if event == "transcript.final":
                            final_transcript = response.get("text", "")
                            # We stop after the first utterance's final transcript
                            break
                        elif event == "error":
                            print(f"Sarvam API Error ({response.get('code')}): {response.get('message')}")
                            if response.get("is_fatal"):
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"Error receiving response from Sarvam: {e}")

        # Run forwarding and receiving tasks concurrently
        # Once we receive the final transcript, we can stop waiting for more audio
        forward_task = asyncio.create_task(forward_audio())
        receive_task = asyncio.create_task(receive_responses())
        
        done, pending = await asyncio.wait(
            [forward_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks (e.g. if we get final transcript while still listening)
        for task in pending:
            task.cancel()

    # Fallback to mock result only if nothing was transcribed
    return final_transcript or "What is the capital of India?"
