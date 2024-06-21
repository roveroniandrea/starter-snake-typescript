import { GameState } from './types';
import { Moves } from './utils';
import * as tf from '@tensorflow/tfjs';

export class SnakeAgent {

    private readonly model: tf.Sequential;
    private readonly learningRate: number;
    private readonly discountFactor: number;
    private readonly movesByIndex: Moves[] = [
        Moves.up,
        Moves.down,
        Moves.left,
        Moves.right,
    ]

    constructor() {
        this.model = this.createModel();
        this.learningRate = 0.1;
        this.discountFactor = 0.9;
    }

    createModel(): tf.Sequential {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [4], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        // Output layer
        model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        return model;
    }


    public async train(prevMove: Moves, newState: GameState): Promise<void> {
        const target = reward + this.discountFactor * Math.max(...this.model.predict(tf.tensor([nextState])).dataSync());
        // So this runs twice?
        const targetF = this.model.predict(tf.tensor([state])).dataSync();
        targetF[action] = target;

        await this.model.fit(tf.tensor([state]), tf.tensor([targetF]), { epochs: 1 });

    }

    public play(gameState: GameState): Moves {
        const qValues = this.model.predict(tf.tensor([state])).dataSync();
        const moveIndex: number = qValues.indexOf(Math.max(...qValues));

        return this.movesByIndex[moveIndex];
    }

    private mapStateToInput(state: GameState): unknown {

    }
}