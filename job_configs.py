import os

EAI_ACCOUNT_ID_TO_USER = {
    "123": "123",
}

eai_account_id = os.environ["EAI_ACCOUNT_ID"]
user = EAI_ACCOUNT_ID_TO_USER[eai_account_id]

JOB_CONFIG = {
    "base": {
        "account_id": eai_account_id,
        "image": "registry.console.elementai.com/snow.colab_public/ssh",
        "data": [
            f"snow.{user}.home:/mnt/home",
            "snow.colab_public.data:/mnt/colab_public",
        ],
        "restartable": True,
        "resources": {"cpu": 8, "mem": 32, "gpu": 4, "gpu_mem": 80},
        "interactive": False,
        "bid": 0,
    },
}
