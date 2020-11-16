import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from mjproto.mj_pb2 import *
from mjproto.mj_pb2_grpc import *