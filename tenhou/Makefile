proto:
	python -m grpc_tools.protoc -I ../protos --python_out=. --grpc_python_out=. ../protos/mj.proto

clean:
	rm mj_pb2.py mj_pb2_grpc.py


.PHONY:	proto clean
