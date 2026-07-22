import sys
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

def inspect_mcap(mcap_path):
    print(f"Inspecting: {mcap_path}")
    topics = {}
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic not in topics:
                    topics[channel.topic] = {
                        "type": schema.name,
                        "count": 0
                    }
                topics[channel.topic]["count"] += 1
        
        print(f"{'Topic':<50} {'Type':<30} {'Count':<10}")
        print("-" * 90)
        for topic, info in topics.items():
            print(f"{topic:<50} {info['type']:<30} {info['count']:<10}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/tools/diagnostics/inspect_mcap.py <mcap_file>")
    else:
        inspect_mcap(sys.argv[1])
