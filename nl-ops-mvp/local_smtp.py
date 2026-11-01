"""Loopback-only SMTP capture service. Run: python local_smtp.py"""
import logging
import threading

from aiosmtpd.controller import Controller

from smtp_inbox import store_message


class CaptureHandler:
    async def handle_DATA(self, server, session, envelope):
        try:
            target = store_message(envelope.original_content)
            logging.info("Captured local test email: %s", target.name)
            return "250 Message captured locally; no external delivery"
        except OSError:
            logging.exception("Local email persistence failed")
            return "451 Could not persist local message"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = Controller(CaptureHandler(), hostname="127.0.0.1", port=1025,
                            data_size_limit=1024 * 1024, ready_timeout=10)
    controller.start()
    print("Local email capture running on 127.0.0.1:1025. View emails in the Streamlit DBA Email Inbox.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
