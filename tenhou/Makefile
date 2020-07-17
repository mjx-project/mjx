proto:
	python -m grpc_tools.protoc -I ../protos --python_out=. --grpc_python_out=. ../protos/mahjong.proto

clean:
	rm mahjong_pb2.py mahjong_pb2_grpc.py


.PHONY:	proto clean
