import threading
import time

# Create two locks (resources)
lock1 = threading.Lock()
lock2 = threading.Lock()

def thread1():
    print("Thread-1 acquiring both locks, performing critical section.")
    
    # Acquire locks in order
    with lock1:
        time.sleep(1)  # simulate delay
        with lock2:
            print("Thread-1 acquired both locks, performing critical section.")
            time.sleep(1)
    
    print("Thread-1 released both locks.")

def thread2():
    print("Thread-2 acquiring both locks, performing critical section.")
    
    # SAME ORDER → prevents deadlock
    with lock1:
        time.sleep(1)
        with lock2:
            print("Thread-2 acquired both locks, performing critical section.")
            time.sleep(1)
    
    print("Thread-2 released both locks.")

# Create threads
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)

# Start threads
t1.start()
t2.start()

# Wait for completion
t1.join()
t2.join()

print("\nBoth threads completed successfully. No deadlock occurred.")