import multiprocessing
import itertools
import time
from osim.env import RunEnv

CPU_COUNT = multiprocessing.cpu_count()


def spawn_process(pid):
    env = RunEnv(visualize=False)
    _ = env.reset(difficulty=0)
    for i in range(200):
        observation, reward, done, info = env.step(env.action_space.sample())


def test_processes():
    processes = []
    for process_idx in range(CPU_COUNT):
        process = multiprocessing.Process(target=spawn_process, args=(process_idx,))
        process.daemon = True
        processes.append(process)

    start = time.time()
    for process in processes:
        process.start()

    for process in processes:
        process.join()
    stop = time.time()
    elapsed = stop - start
    aps = (len(processes) * 200) / elapsed
    print("Processes APS: ", aps)


def test_pool():
    with multiprocessing.Pool(CPU_COUNT) as pool:
        start = time.time()
        _ = pool.map(spawn_process, itertools.count())
    stop = time.time()
    elapsed = stop - start
    aps = (CPU_COUNT * 200) / elapsed
    print("Processes APS: ", aps)


test_pool()
