# 使用 Python 3.11 和 Alpine 作为基础镜像
FROM python:3.11-alpine

WORKDIR /app

# 安装系统依赖
RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    openssl-dev

# 复制应用程序文件
COPY requirements.txt .
COPY vanna-mysql.py .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8084

# 启动应用
CMD ["python", "vanna-mysql.py"]