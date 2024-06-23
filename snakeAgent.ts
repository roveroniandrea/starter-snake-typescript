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
    ];

    private prevTargetQValues: number[] | null = null;
    private prevStateTensor: tf.Tensor | null = null;
    private prevMove: Moves | null = null;

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


    public async train(newState: GameState): Promise<void> {
        if (!this.prevTargetQValues || !this.prevStateTensor || !this.prevMove) {
            throw new Error("Missing previous state");
        }

        // TODO: Per evitare race conditions, per ora è sincrono (circa, andrà fatto un qualche semaforo)
        // TODO: Calc reward
        const reward: number = 0;
        // 1. Prevediamo i valori Q per lo stato successivo
        const nextStateTensor = this.mapStateToInput(newState);
        const nextQValues = this.model.predict(nextStateTensor);
        const maxNextQValue = Math.max(...(nextQValues as tf.Tensor).dataSync());

        // 2. Calcoliamo il target Q-value per l'azione presa
        const target = reward + this.discountFactor * maxNextQValue;

        // 3. Prevediamo i valori Q per lo stato attuale (riutilizzato da calcolo precedente)

        // 4. Sostituiamo il valore Q corrispondente all'azione presa con il target calcolato
        const action: number = parseInt(Object.keys(this.movesByIndex).find(i => this.movesByIndex[parseInt(i)] === this.prevMove) || '');

        const targetQValues: number[] = this.prevTargetQValues.slice();
        targetQValues[action] = target;

        // 5. Eseguiamo un passo di addestramento per il modello
        const targetTensor = tf.tensor([targetQValues]);
        await this.model.fit(this.prevStateTensor, targetTensor, { epochs: 1 });

        // 6. Pulizia delle variabili tensor
        nextStateTensor.dispose();
        this.prevStateTensor.dispose();
        targetTensor.dispose();

        this.prevTargetQValues = null;
        this.prevStateTensor = null;
        this.prevMove = null;

    }

    public async play(gameState: GameState): Promise<Moves> {
        if (this.prevTargetQValues || this.prevStateTensor || this.prevMove) {
            throw new Error("Not trained on previous state");
        }

        const newStateTensor: tf.Tensor = this.mapStateToInput(gameState);

        const qValues = await (this.model.predict(newStateTensor) as tf.Tensor<tf.Rank>).data();
        const moveIndex: number = qValues.indexOf(Math.max(...qValues));

        const move: Moves = this.movesByIndex[moveIndex];

        this.prevStateTensor = newStateTensor;
        this.prevTargetQValues = [...qValues];
        this.prevMove = move;

        return move;
    }

    private mapStateToInput(state: GameState): tf.Tensor {
        // TODO:
        // Reshape allows to set the correct dimension to the tensor
        return tf.tensor([1, 0, 0, 0]).reshape([-1, 4]);
    }
}