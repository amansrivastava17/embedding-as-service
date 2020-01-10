from typing import Union, List, Optional
import numpy as np
import zmq
import json


class EmbeddingClient(object):
    """
    Represents an example client.
    """
    def __init__(self, host, port):
        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.DEALER)
        self.socket.connect(f'tcp://{host}:{port}')
        self.identity = '123'

    def encode(self,
               texts: Union[List[str], List[List[str]]],
               pooling: Optional[str] = None,
               is_tokenized: bool = False,
               batch_size: int = 256,
               **kwargs
               ) -> np.array:
        """
        Connects to server. Send compute request, poll for and print result to standard out.
        """
        if not isinstance(texts, list):
            raise ValueError('Argument `texts` should be either List[str] or List[List[str]]')
        if is_tokenized:
            if not all(isinstance(text, list) for text in texts):
                raise ValueError('Argument `texts` should be List[List[str]] (list of tokens) '
                                 'when `is_tokenized` = True')
        embeddings = []
        for i in range(0, len(texts), batch_size):
            request_data = {
                'type': 'encode',
                'texts': texts[i: i+batch_size],
                'pooling': pooling,
                'is_tokenized': is_tokenized
            }
            self.send(json.dumps(request_data))
            result = self.receive()
            result = json.loads(result.decode("utf-8"))
            embeddings.append(np.array(result))
        embeddings = np.vstack(embeddings)
        return embeddings

    def terminate(self):
        self.socket.close()
        self.zmq_context.term()

    def send(self, data):
        """
        Send data through provided socket.
        """
        self.socket.send_string(data)

    def receive(self):
        """
        Receive and return data through provided socket.
        """
        return self.socket.recv()

    def tokenize(self, texts: Union[List[str], str]) -> np.array:
        request_data = {
            'type': 'tokenize',
            'texts': texts
        }

        self.send(json.dumps(request_data))
        result = self.receive()
        result = json.loads(result.decode("utf-8"))
        return result['tokens']

