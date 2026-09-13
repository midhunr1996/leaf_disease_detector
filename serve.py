"""Local web server for the tea leaf inspection interface.

Run with:  python serve.py [port]

The plain `python -m http.server` works, but it sends no cross-origin isolation
headers. Without those the browser refuses to start WebAssembly worker threads,
so ONNX Runtime falls back to a single thread and inference crawls. The two
headers below turn isolation on, which lets the runtime use every CPU core.

`credentialless` is used rather than `require-corp` so the ONNX Runtime bundle
still loads from the CDN, which does not set a CORP header of its own.
"""

import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
HOST = "127.0.0.1"


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = dict(
        http.server.SimpleHTTPRequestHandler.extensions_map,
        **{
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".wasm": "application/wasm",
            ".onnx": "application/octet-stream",
        }
    )

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # The model is large and rarely changes, but the page and script are
        # edited often, so keep those out of the cache while developing.
        if self.path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    with Server((HOST, PORT), Handler) as httpd:
        print("Tea leaf inspection interface: http://%s:%d/index.html" % (HOST, PORT))
        print("Cross-origin isolation is on, so WASM threads are available.")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
