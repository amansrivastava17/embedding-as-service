from embedding_as_service.text.encode import Encoder

from typing import Union, List, Optional, Dict
import threading
import argparse
import zmq
import json
import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(level=logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Server(object):

    def __init__(self, embedding, model, max_seq_length, port, num_workers=4):
        self.zmq_context = zmq.Context()
        self.port = port
        self.num_workers = num_workers
        self.encoder = Encoder(embedding=embedding, model=model, max_seq_length=max_seq_length)

    def start(self):
        """
        Main execution.
        Instantiate workers, Accept client connections,
        distribute computation requests among workers and route computed results back to clients.
        """

        # Front facing socket to accept client connections.
        socket_front = self.zmq_context.socket(zmq.ROUTER)
        socket_front.bind(f'tcp://0.0.0.0:{self.port}')

        # Backend socket to distribute work.
        socket_back = self.zmq_context.socket(zmq.DEALER)
        socket_back.bind('inproc://backend')

        # Start workers.
        for i in range(0, self.num_workers):
            worker = Worker(self.zmq_context, self.encoder, i)
            worker.start()
            logger.info(f'[WORKER-{i}]: ready and listening!')

        # Use built in queue device to distribute requests among workers.
        # What queue device does internally is,
        #   1. Read a client's socket ID and request.
        #   2. Send socket ID and request to a worker.
        #   3. Read a client's socket ID and result from a worker.
        #   4. Route result back to the client using socket ID.
        zmq.device(zmq.QUEUE, socket_front, socket_back)


class Worker(threading.Thread):
    """
    Workers accept computation requests from front facing server.
    Does computations and return results back to server.
    """

    def __init__(self, zmq_context, encoder, _id):
        threading.Thread.__init__(self)
        self.zmq_context = zmq_context
        self.worker_id = _id
        self.encoder = encoder

    def run(self):
        """
        Main execution.
        Returns:
        """
        # Socket to communicate with front facing server.
        socket = self.zmq_context.socket(zmq.DEALER)
        socket.connect('inproc://backend')

        while True:
            # First string recieved is socket ID of client
            client_id = socket.recv()
            request = socket.recv()
            # print('Worker ID - %s. Recieved computation request.' % (self.worker_id))

            result = self.compute(request)

            # print('Worker ID - %s. Sending computed result back.' % (self.worker_id))

            # For successful routing of result to correct client, the socket ID of client should be sent first.
            socket.send(client_id, zmq.SNDMORE)
            socket.send_string(result)

    def compute(self, request):
        """Computation takes place here. Adds the two numbers which are in the request and return result."""
        request = json.loads(request.decode("utf-8"))
        _type = request.get('type')
        if _type == 'encode':
            return self.encode(request)
        elif _type == 'tokenize':
            return self.tokenize(request)
        return

    def encode(self, data):
        texts: Union[List[str], List[List[str]]] = data['texts']
        pooling: Optional[str] = data.get('pooling')
        is_tokenized: bool = data.get('is_tokenized', False)
        batch_size: int = data.get('batch_size', 256)
        embedding = self.encoder.encode(
            texts=texts,
            pooling=pooling,
            is_tokenized=is_tokenized,
            batch_size=batch_size
        )
        return json.dumps(embedding.tolist())

    def tokenize(self, data):
        texts: Union[List[str], List[List[str]]] = data.get('texts')
        tokens = self.encoder.tokenize(texts)
        return json.dumps({'tokens': tokens})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', required=True, type=str, help='name of the embedding type')
    parser.add_argument('--model', required=True, type=str, help='name of the embedding model')
    parser.add_argument('--max_seq_length', required=False, default=128, type=int, help='max sequence length')
    parser.add_argument('--port', required=False, type=int, default=8080, help='port to run on')
    parser.add_argument('--num_workers', required=False, type=int, default=4, help='number of workers on the server')
    args = parser.parse_args()
    server = Server(embedding=args.embedding, model=args.model, max_seq_length=args.max_seq_length,
                    port=args.port, num_workers=args.num_workers)
    server.start()
