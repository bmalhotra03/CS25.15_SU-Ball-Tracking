from gopro_manager import GoProManager
import signal
import time

def main():
    try: 
        manager = GoProManager(ip_addresses=[])
        signal.signal(signal.SIGINT, lambda s, f: manager.kill_stream_controller())
        manager.start_processes()
        while manager.continue_stream:
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting")


if __name__ == "__main__":
    main()