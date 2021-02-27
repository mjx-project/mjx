import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from mjproto.mjx_pb2_grpc import *
from mjproto.mjx_pb2 import *