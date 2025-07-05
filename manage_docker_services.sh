#!/bin/bash
# Friday AI Trading System - Docker Services Management Script
# This script manages MongoDB and Redis Docker containers

set -e

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="friday"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    echo -e "$1"
}

show_help() {
    echo
    print_message "${BLUE}Friday AI Trading System - Docker Services Management${NC}"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start MongoDB and Redis services"
    echo "  stop      Stop MongoDB and Redis services"
    echo "  restart   Restart MongoDB and Redis services"
    echo "  status    Show status of services"
    echo "  logs      Show logs from services"
    echo "  clean     Stop and remove containers, networks, and volumes"
    echo "  setup     Create required directories and start services"
    echo "  health    Check health status of services"
    echo "  help      Show this help message"
    echo
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_message "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker and ensure it's running"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_message "${RED}Error: Docker daemon is not running${NC}"
        echo "Please start Docker daemon"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_message "${RED}Error: docker-compose is not installed${NC}"
        echo "Please install docker-compose"
        exit 1
    fi
}

create_directories() {
    print_message "${BLUE}Creating required directories...${NC}"
    mkdir -p storage/data/mongodb
    mkdir -p storage/data/redis
    mkdir -p storage/data/mongodb/configdb
    print_message "${GREEN}Directories created successfully${NC}"
}

start_services() {
    print_message "${BLUE}Starting Friday Docker services...${NC}"
    if ! docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d; then
        print_message "${RED}Failed to start services${NC}"
        exit 1
    fi
    print_message "${GREEN}Services started successfully${NC}"
    echo
    print_message "${YELLOW}Waiting for services to be ready...${NC}"
    python3 wait_for_services.py || python wait_for_services.py
}

stop_services() {
    print_message "${BLUE}Stopping Friday Docker services...${NC}"
    if ! docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down; then
        print_message "${RED}Failed to stop services${NC}"
        exit 1
    fi
    print_message "${GREEN}Services stopped successfully${NC}"
}

restart_services() {
    print_message "${BLUE}Restarting Friday Docker services...${NC}"
    stop_services
    sleep 3
    start_services
}

show_status() {
    print_message "${BLUE}Friday Docker Services Status:${NC}"
    echo
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    echo
    print_message "${BLUE}Container Health Status:${NC}"
    docker inspect friday_mongodb --format="MongoDB: {{.State.Health.Status}}" 2>/dev/null || echo "MongoDB: Not running"
    docker inspect friday_redis --format="Redis: {{.State.Health.Status}}" 2>/dev/null || echo "Redis: Not running"
}

show_logs() {
    print_message "${BLUE}Friday Docker Services Logs:${NC}"
    if [ -z "$2" ]; then
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
    else
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$2"
    fi
}

clean_services() {
    print_message "${YELLOW}This will stop and remove all Friday Docker containers, networks, and volumes${NC}"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "${BLUE}Cleaning up Friday Docker environment...${NC}"
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v --remove-orphans
        print_message "${GREEN}Cleanup completed${NC}"
    else
        print_message "${YELLOW}Cleanup cancelled${NC}"
    fi
}

setup_environment() {
    print_message "${BLUE}Setting up Friday Docker environment...${NC}"
    create_directories
    start_services
    print_message "${GREEN}Setup completed successfully${NC}"
}

check_health() {
    print_message "${BLUE}Checking health of Friday services...${NC}"
    python3 wait_for_services.py --timeout 30 || python wait_for_services.py --timeout 30
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

check_docker

# Command processing
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$@"
        ;;
    clean)
        clean_services
        ;;
    setup)
        setup_environment
        ;;
    health)
        check_health
        ;;
    help)
        show_help
        ;;
    *)
        print_message "${RED}Unknown command: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
