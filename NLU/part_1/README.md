To run the file:

```python
python3 main.py <model> <optmizer>
```

The available models are:
- `RNN`
	- optimizer `SGD`
- `LSTM`  
	- optimizer `RNN`
- `LSTM_dropout`
	1. optimizer `SGD`
    2. optimizer `AdamW`

So we possible combinations are:

```python
1. python3 main.py RNN SGD
2. python3 main.py LSTM SGD
3. python3 main.py LSTM_dropout SGD
4. python3 main.py LSTM_dropout AdamW
```

The program will automatically adapt the parameters based on the model / optimizer chosen. 
