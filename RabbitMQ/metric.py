import pika
import pickle
import json
import numpy as np
import nyoka
import csv
import os.path
import time

from sklearn.metrics import mean_squared_error as mse


class Consumer:

    header = ['id', 'y_true', 'y_pred', 'rmse']
    writer = None

    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        self.features = None
        self.y_true = 0
        self.y_pred = 0
        self.r_mse = 0
        self.metrics = []
        self.true_labels = np.empty(0)
        self.pred_labels = np.empty(0)
        self.id = None
        self.result_file = open('result.csv', 'a+')
        self.result_file.truncate()

        with open('pipe.pkl', 'rb') as f:
            self.model = pickle.load(f)

        print('instance created')

    def get_y_pred(self, ch, method, props, body):
        params = json.loads(body)
        self.id = params['id']
        self.features = np.array(params['X_features']).reshape(1, -1)
        self.y_pred = self.model.predict(self.features)

    def get_y_true(self, ch, method, props, body):
        params = json.loads(body)
        self.y_true = np.array([params['y_true']])
        print('y_true = {}'.format(self.y_true))
        self.write_result_to_file()

    def write_result_to_file(self):
        if Consumer.writer is None:
            Consumer.writer = csv.DictWriter(self.result_file, delimiter=';', lineterminator='\n', fieldnames=Consumer.header)

        #metric = np.sqrt(mse(self.y_true, self.y_pred))
        #self.metrics.append(metric)

        self.pred_labels = np.append(self.pred_labels, self.y_pred)
        self.true_labels = np.append(self.true_labels, self.y_true)

        row = {
            'id': self.id,
            'y_true': self.y_true[0],
            'y_pred': self.y_pred[0],
            'rmse': np.sqrt(mse(self.true_labels, self.pred_labels))
        }

        try:
            Consumer.writer.writerow(row)
            print('row written!')
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except Exception as ex:
            print("Unexpected error({0}): {1}".format(ex.errno, ex.strerror))

    def processing(self):
        try:
            self.channel.queue_declare(queue='X_features')
            self.channel.queue_declare(queue='y_true')
            self.channel.basic_consume(queue='X_features', on_message_callback=self.get_y_pred, auto_ack=True)
            self.channel.basic_consume(queue='y_true', on_message_callback=self.get_y_true, auto_ack=True)
            self.channel.start_consuming()

        except KeyboardInterrupt:
            self.connection.close()
            self.result_file.close()
            print('Consuming stopped')


if __name__ == '__main__':
    consumer = Consumer()
    consumer.processing()

