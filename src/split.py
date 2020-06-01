#!/usr/bin/env python

import pandas
import click
from sklearn.model_selection import train_test_split

@click.command()
@click.option("--train-size",default=0.8,help="Size of the training set")
@click.argument('data')
@click.argument('train-path')
@click.argument('test-path')
def main(data,train_path,test_path,train_size):
  """Split a csv file into random train and test subsets and export it."""
  df = pandas.read_csv(data)

  train,test=train_test_split(df,train_size=train_size,random_state =3 )


 # Export csv files
  train.to_csv(train_path,index = False)
  test.to_csv(test_path,index = False)

  for f in [train_path,test_path]:
      click.echo("The file {} was written. ".format(f))

if __name__ == "__main__":
    main()
