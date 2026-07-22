#!/usr/bin/env bash
set +u 

XML="${XML:-/home/ubuntu/eai-data-platform-web/config/client.xml}"
PEERS="${PEERS:-}"
DOMAIN="${DOMAIN:-0}"
GENERATE_XML="${GENERATE_XML:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --xml) XML="$2"; shift 2;;
    --peers) PEERS="$2"; shift 2;;
    --domain) DOMAIN="$2"; shift 2;;
    --generate-xml) GENERATE_XML="1"; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "Missing /opt/ros/humble/setup.bash"; exit 1
fi

source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$XML"
export ROS_DOMAIN_ID="$DOMAIN"
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
export ROS2_DISABLE_DAEMON=1

if [ ! -f "$XML" ] && [ "$GENERATE_XML" = "1" ]; then
  if [ -z "$PEERS" ]; then
    echo "PEERS required to generate XML"; exit 1
  fi
  mkdir -p "$(dirname "$XML")"
  {
    echo '<?xml version="1.0" encoding="UTF-8" ?>'
    echo '<dds>'
    echo '  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">'
    echo '    <participant profile_name="peer_participant" is_default_profile="true">'
    echo '      <rtps>'
    echo '        <builtin>'
    echo '          <discovery_config>'
    echo '            <discoveryProtocol>SIMPLE</discoveryProtocol>'
    echo '            <initialAnnouncements>'
    echo '              <count>10</count>'
    echo '              <period><sec>1</sec><nanosec>0</nanosec></period>'
    echo '            </initialAnnouncements>'
    echo '          </discovery_config>'
    echo '          <initialPeersList>'
    IFS=',' read -r -a arr <<< "$PEERS"
    for ip in "${arr[@]}"; do
      echo "            <locator><udpv4><address>${ip}</address></udpv4></locator>"
    done
    echo '          </initialPeersList>'
    echo '        </builtin>'
    echo '      </rtps>'
    echo '    </participant>'
    echo '  </profiles>'
    echo '</dds>'
  } > "$XML"
fi

ros2 daemon stop || true

echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
echo "ROS_DISCOVERY_SERVER=${ROS_DISCOVERY_SERVER-}"
echo "ROS2_DISABLE_DAEMON=$ROS2_DISABLE_DAEMON"
echo "XML_EXISTS=$( [ -f "$XML" ] && echo yes || echo no )"

ros2 node list || true
ros2 topic list || true
