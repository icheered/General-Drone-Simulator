import time

start_time = time.time()

time.sleep(1)

print(time.time() - start_time)
# Cast to shape hh:mm:ss
print(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))