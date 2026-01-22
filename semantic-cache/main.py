import os
import redis

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')


def main():
    r = redis.Redis.from_url(REDIS_URL)
    try:
        r.ping()
        print("✅ Connected to Redis successfully!")
    except redis.ConnectionError:
        print(f"❌ Failed to connect to Redis at {REDIS_URL}")


if __name__ == "__main__":
    main()
