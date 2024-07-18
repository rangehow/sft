import subprocess


def get_shm_info():
    # 运行 df -h /dev/shm 命令
    result = subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True)

    # 检查命令是否成功执行
    if result.returncode != 0:
        raise RuntimeError("Failed to run df command")

    # 解析命令输出
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        raise RuntimeError("Unexpected df command output")

    # 提取相关信息
    header = lines[0].split()
    values = lines[1].split()

    # 将结果存储在字典中
    shm_info = dict(zip(header, values))

    return int(shm_info["Size"][:-1])
