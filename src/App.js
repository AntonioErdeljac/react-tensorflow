import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import * as tf from '@tensorflow/tfjs';

class App extends Component {
  constructor(props) {
    super(props);

    this.state = {
      inputValue: '',
      prediction: '',
    };

    this.linearModel = tf.Sequential;
    this.linearPrediction = this.linearPrediction.bind(this);
    this.trainNewModel = this.trainNewModel.bind(this);
  }

  componentDidMount() {
    this.trainNewModel();
  }

  linearPrediction(value) {
    const output = this.linearModel.predict(tf.tensor2d([value], [1, 1]));

    console.log(output);

    this.setState({
      prediction: Array.from(output.dataSync())[0],
      inputValue: value
    });

  }

  async trainNewModel() {
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({
      units: 1,
      inputShape: [1]
    }));

    this.linearModel.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd',
    });

    const xs = tf.tensor1d([3.2, 4.4, 5.5]);
    const ys = tf.tensor1d([1.6, 2.7, 3.5]);

    await this.linearModel.fit(xs, ys);
  }

  render() {
    const { inputValue, prediction } = this.state;

    return (
      <div className="container">
        <div className="row">
          <div className="col-12">
            <h1>Linear model</h1>
            <b>{prediction}</b>
          </div>
          <div className="col-md-4 col-12">
            <input className="form-control" value={inputValue} onChange={(ev) => this.linearPrediction(ev.target.value)} />
          </div>
        </div>
      </div>
    );
  }
}

export default App;
