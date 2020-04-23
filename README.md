# DSLR

School 42 Data Science Project 

About
-----
Code the Sorting Hat of Hogwarts school


Usage
-----
`python3 describe.py dataset_path`
* Display features informations

`python3 histogram.py dataset_path`
* Answer the following question: Which Hogwarts course has a homogeneous score distribution between the four houses ?

`python3 scatter_plot.py dataset_path`
* Answer the following question: Which features are similar ?

`python3 pair_plot.py dataset_path`
* Show a pair_plot of all features present

`logreg_train.py [-h] [-s] [-a] [-c] [-p] [-f] [-e E] [-ls] [-log] [-v] file`

positional arguments:
  file        input dataset path

optional arguments:
  -h, --help  show this help message and exit
  -s          Stochastic gradient descent
  -a          Stochastic gradient descent with adam optimizer
  -c          Check results
  -p          Plot cost function
  -f          Forward fill method for NaN values. Default: DropNaN
  -e int        Epochs Number. Default: 100
  -ls         Linear Scalling. Default : Z_score
  -log        Log Scalling. Default : Z_score
  -v          Show compare

  `python3 logreg_predict.py dataset weights`
* Generate a file with all predictions for a given dataset

###Example :

`logreg_train.py -s -c -e 50 file`
