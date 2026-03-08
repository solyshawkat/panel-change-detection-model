"""
Eureka service registration.
Registers PCD-AI with Eden's Eureka server so the backend can discover us.
"""
import logging
import socket
from app.core import config

logger = logging.getLogger(__name__)

_client = None


def get_instance_ip() -> str:
    """Get the machine's IP address for Eureka registration."""
    if config.EUREKA["instance_host"]:
        return config.EUREKA["instance_host"]
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


async def register():
    """Register with Eureka on startup."""
    if not config.EUREKA["enabled"]:
        logger.info("Eureka registration disabled")
        return

    try:
        import py_eureka_client.eureka_client as eureka_client

        instance_ip = get_instance_ip()

        await eureka_client.init_async(
            eureka_server=config.EUREKA["server_url"],
            app_name=config.APP_NAME,
            instance_host=instance_ip,
            instance_port=config.APP_PORT,
            health_check_url=f"http://{instance_ip}:{config.APP_PORT}{config.CONTEXT_PATH}/health",
            home_page_url=f"http://{instance_ip}:{config.APP_PORT}{config.CONTEXT_PATH}/",
            status_page_url=f"http://{instance_ip}:{config.APP_PORT}{config.CONTEXT_PATH}/docs",
        )

        logger.info(
            f"Registered with Eureka: {config.APP_NAME} "
            f"at {instance_ip}:{config.APP_PORT}"
        )

    except ImportError:
        logger.warning("py_eureka_client not installed. Skipping Eureka registration.")
    except Exception as e:
        logger.error(f"Eureka registration failed: {e}")
        # Don't crash the service — it can still work without Eureka


async def deregister():
    """Deregister from Eureka on shutdown."""
    if not config.EUREKA["enabled"]:
        return

    try:
        import py_eureka_client.eureka_client as eureka_client
        await eureka_client.stop_async()
        logger.info("Deregistered from Eureka")
    except Exception as e:
        logger.warning(f"Eureka deregistration failed: {e}")
