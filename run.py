from neural import Perceptron

perceptron = Perceptron(filepath='data.txt')
perceptron.train()
perceptron.predict(filepath='predict.txt')

