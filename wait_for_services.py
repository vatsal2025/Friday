#!/usr/bin/env python3
"""
Health check utility for Friday AI Trading System services.
This script waits for MongoDB and Redis services to be ready before proceeding.
"""

import time
import sys
import argparse
import logging
from typing import List, Dict, Any
import subprocess
import socket

# Import configuration
try:
    from unified_config import MONGODB_CONFIG, REDIS_CONFIG, CACHE_CONFIG
except ImportError:
    # Fallback configuration if unified_config is not available
    MONGODB_CONFIG = {"host": "localhost", "port": 27017}
    REDIS_CONFIG = {"host": "localhost", "port": 6379}
    CACHE_CONFIG = {"redis_host": "localhost", "redis_port": 6379}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceHealthChecker:
    """Health checker for external services."""
    
    def __init__(self, timeout: int = 300):
        """
        Initialize the health checker.
        
        Args:
            timeout: Maximum time to wait for all services (seconds)
        """
        self.timeout = timeout
        self.start_time = time.time()
    
    def check_port_open(self, host: str, port: int, timeout: int = 5) -> bool:
        """
        Check if a port is open and accepting connections.
        
        Args:
            host: Host to check
            port: Port to check
            timeout: Connection timeout
            
        Returns:
            True if port is open, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed for {host}:{port} - {e}")
            return False
    
    def check_mongodb_health(self) -> bool:
        """
        Check MongoDB health using mongosh.
        
        Returns:
            True if MongoDB is healthy, False otherwise
        """
        try:
            # First check if port is open
            mongo_host = MONGODB_CONFIG.get("host", "localhost")
            mongo_port = MONGODB_CONFIG.get("port", 27017)
            
            if not self.check_port_open(mongo_host, mongo_port):
                logger.debug(f"MongoDB port {mongo_port} not accessible")
                return False
            
            # Try to ping MongoDB using mongosh
            cmd = [
                "mongosh", 
                f"mongodb://{mongo_host}:{mongo_port}/admin",
                "--eval", 
                "db.adminCommand('ping')",
                "--quiet"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and "ok" in result.stdout.lower():
                logger.debug("MongoDB health check passed")
                return True
            else:
                logger.debug(f"MongoDB health check failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.debug("MongoDB health check timed out")
            return False
        except FileNotFoundError:
            logger.warning("mongosh not found, falling back to port check")
            # Fallback to port check only
            return self.check_port_open(mongo_host, mongo_port)
        except Exception as e:
            logger.debug(f"MongoDB health check error: {e}")
            return False
    
    def check_redis_health(self) -> bool:
        """
        Check Redis health using redis-cli.
        
        Returns:
            True if Redis is healthy, False otherwise
        """
        try:
            # First check if port is open
            redis_host = REDIS_CONFIG.get("host", CACHE_CONFIG.get("redis_host", "localhost"))
            redis_port = REDIS_CONFIG.get("port", CACHE_CONFIG.get("redis_port", 6379))
            
            if not self.check_port_open(redis_host, redis_port):
                logger.debug(f"Redis port {redis_port} not accessible")
                return False
            
            # Try to ping Redis using redis-cli
            cmd = ["redis-cli", "-h", redis_host, "-p", str(redis_port), "ping"]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and "pong" in result.stdout.lower():
                logger.debug("Redis health check passed")
                return True
            else:
                logger.debug(f"Redis health check failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.debug("Redis health check timed out")
            return False
        except FileNotFoundError:
            logger.warning("redis-cli not found, falling back to port check")
            # Fallback to port check only
            return self.check_port_open(redis_host, redis_port)
        except Exception as e:
            logger.debug(f"Redis health check error: {e}")
            return False
    
    def wait_for_services(self, services: List[str]) -> bool:
        """
        Wait for specified services to be ready.
        
        Args:
            services: List of service names to wait for
            
        Returns:
            True if all services are ready, False if timeout
        """
        service_checkers = {
            "mongodb": self.check_mongodb_health,
            "redis": self.check_redis_health
        }
        
        logger.info(f"Waiting for services: {', '.join(services)}")
        logger.info(f"Timeout set to {self.timeout} seconds")
        
        while time.time() - self.start_time < self.timeout:
            all_ready = True
            service_status = {}
            
            for service in services:
                if service not in service_checkers:
                    logger.error(f"Unknown service: {service}")
                    return False
                
                is_ready = service_checkers[service]()
                service_status[service] = is_ready
                
                if not is_ready:
                    all_ready = False
            
            if all_ready:
                logger.info("✅ All services are ready!")
                return True
            
            # Log current status
            status_msgs = []
            for service, is_ready in service_status.items():
                status = "✅" if is_ready else "❌"
                status_msgs.append(f"{service}: {status}")
            
            elapsed = int(time.time() - self.start_time)
            logger.info(f"[{elapsed}s] {' | '.join(status_msgs)}")
            
            time.sleep(5)  # Wait 5 seconds before next check
        
        logger.error(f"❌ Timeout after {self.timeout} seconds waiting for services")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Wait for Friday AI Trading System services to be ready"
    )
    parser.add_argument(
        "services",
        nargs="*",
        default=["mongodb", "redis"],
        help="Services to wait for (default: mongodb redis)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate service names
    valid_services = ["mongodb", "redis"]
    for service in args.services:
        if service not in valid_services:
            logger.error(f"Invalid service: {service}")
            logger.error(f"Valid services: {', '.join(valid_services)}")
            sys.exit(1)
    
    # Create health checker and wait for services
    checker = ServiceHealthChecker(timeout=args.timeout)
    
    try:
        if checker.wait_for_services(args.services):
            logger.info("🎉 All services are ready! System can proceed.")
            sys.exit(0)
        else:
            logger.error("💥 Services not ready within timeout period")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⏹️  Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
