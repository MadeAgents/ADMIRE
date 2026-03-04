python src/hammer_server/gradio_web_server.py \
        --num-devices 8 \
        --max-devices 8 \
        --crashed-device-restart \
        --concurrency-limit 8 \
        --state-waiting 2 \
        --show-error