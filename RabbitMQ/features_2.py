import pika
import json
import numpy as np
import time
import uuid
from sklearn.datasets import load_diabetes


class Publisher:

    DELAY = 6

    def __init__(self):
        self.X, self.y = load_diabetes(return_X_y=True)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

    def get_random_row(self):
        return np.random.randint(0, self.X.shape[0]-1)

    def publish_message(self):
        self.channel.queue_declare(queue='X_features')
        self.channel.queue_declare(queue='y_true')

        try:
            while True:
                random_row = self.get_random_row()
                random_id = str(uuid.uuid4())

                features_body = {
                    'id': random_id,
                    'X_features': list(self.X[random_row])
                }
                y_body = {
                    'id': random_id,
                    'y_true': self.y[random_row]
                }

                self.channel.basic_publish(exchange='', routing_key='X_features', body=json.dumps(features_body))
                self.channel.basic_publish(exchange='', routing_key='y_true', body=json.dumps(y_body))
                print('sent!')
                time.sleep(Publisher.DELAY)

        except KeyboardInterrupt:
            self.connection.close()
            print('Publishing stopped')


if __name__ == '__main__':
    Publisher().publish_message()
