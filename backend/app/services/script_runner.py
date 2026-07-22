import asyncio
import os
import signal
import logging
from typing import List, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ScriptRunner:
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.active_websockets: List[WebSocket] = []
        self.log_history: List[str] = []  # Keep some history for new connections

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.append(websocket)
        # Send history
        for line in self.log_history:
            try:
                await websocket.send_text(line)
            except Exception:
                pass

    def clear_history(self):
        self.log_history = []

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)

    async def broadcast(self, message: str):
        # Add to history (limit to last 1000 lines)
        self.log_history.append(message)
        if len(self.log_history) > 1000:
            self.log_history.pop(0)
            
        to_remove = []
        for connection in self.active_websockets:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                to_remove.append(connection)
        
        for conn in to_remove:
            self.disconnect(conn)

    async def start_script(self, script_path: str, args: List[str] = [], env: Optional[dict] = None):
        if self.process:
            # Check if process is really running
            if self.process.returncode is None:
                await self.broadcast("Warning: Script is already running. Stopping old process...\n")
                try:
                    # Kill the process group
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    # Wait a bit for it to die
                    for _ in range(10): # Wait up to 1 second
                        if self.process.returncode is not None:
                            break
                        await asyncio.sleep(0.1)
                except ProcessLookupError:
                    pass # Process already gone
                except Exception as e:
                    logger.error(f"Failed to kill process: {e}")
                    await self.broadcast(f"Error killing old process: {e}\n")
                
                # Double check
                if self.process.returncode is None:
                     # Force manual cleanup of the object if it's stuck
                     self.process = None

        try:
            # Clear history on new run
            self.log_history = []
            
            if not os.path.exists(script_path):
                await self.broadcast(f"Error: Script file not found at {script_path}\n")
                return False

            # Ensure script is executable
            os.chmod(script_path, 0o755)
            
            cmd = [script_path] + args
            await self.broadcast(f"Starting script: {' '.join(cmd)}\n")
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy() | (env or {}),
                preexec_fn=os.setsid  # Create a new process group so we can kill the whole group
            )
            
            # Start reading output in background tasks
            asyncio.create_task(self._read_stream(self.process.stdout))
            asyncio.create_task(self._read_stream(self.process.stderr))
            
            # Wait for process to finish
            # Pass the process object to avoid race condition if self.process is overwritten
            asyncio.create_task(self._wait_process(self.process))
            
            return True
        except Exception as e:
            await self.broadcast(f"Error starting script: {str(e)}\n")
            logger.exception("Failed to start script")
            return False

    async def stop_script(self):
        if not self.process or self.process.returncode is not None:
            await self.broadcast("Script is not running.\n")
            return False

        try:
            await self.broadcast("Sending SIGINT (Ctrl+C) to script...\n")
            # Send SIGINT to the process group
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGINT)

            # Wait briefly for graceful shutdown.
            # Some scripts may ignore SIGINT (or spawn subprocesses that outlive it),
            # which would keep recording and grow mcap even after UI "stop".
            for _ in range(30):  # ~3s
                if self.process.returncode is not None:
                    break
                await asyncio.sleep(0.1)

            # If still running, force kill.
            if self.process.returncode is None:
                await self.broadcast("SIGINT did not stop script in time; sending SIGKILL...\n")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    # Process group already gone
                    pass

                for _ in range(20):  # ~2s
                    if self.process.returncode is not None:
                        break
                    await asyncio.sleep(0.1)

            ok = self.process.returncode is not None
            if ok:
                await self.broadcast("Script stopped.\n")
                self.process = None
            else:
                await self.broadcast("Failed to stop script.\n")
            return ok
        except Exception as e:
            await self.broadcast(f"Error stopping script: {str(e)}\n")
            return False

    async def _read_stream(self, stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            message = line.decode(errors='replace').strip()
            if message:
                await self.broadcast(f"{message}")

    async def _wait_process(self, process):
        if process:
            await process.wait()
            await self.broadcast(f"\nScript finished with exit code {process.returncode}\n")

script_runner = ScriptRunner()
