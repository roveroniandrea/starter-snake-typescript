import * as tf from '@tensorflow/tfjs';
import { Coord, GameState } from './types';
import { Moves } from './utils';

type TurnData = {
    targetQValues: number[];
    stateTensor: tf.Tensor;
    move: Moves;
    isMoveValid: boolean;
    health: number;
    turn: number;
};

export class SnakeAgent {

    private readonly model: tf.Sequential;
    private readonly discountFactor: number = 0.9;
    private readonly movesByIndex: Moves[] = [
        Moves.up,
        Moves.down,
        Moves.left,
        Moves.right,
    ];
    private readonly inputShape: number = 11 * 11;

    private prevGameDatas: Map<number, TurnData> = new Map();

    constructor() {
        this.model = this.createModel();
    }

    createModel(): tf.Sequential {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [this.inputShape], activation: 'relu' }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        // Output layer
        model.add(tf.layers.dense({ units: 4, activation: 'linear' }));
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
        return model;
    }

    public async train(turnData: TurnData, nextTurnData: TurnData): Promise<{
        reward: number;
    }> {
        const {
            targetQValues: prevTargetQValues,
            stateTensor: prevStateTensor,
            move: prevMove,
            isMoveValid,
            health
        } = turnData;

        const reward: number = !isMoveValid ? -1 :
            (nextTurnData.health > health) ? 1 : 0.5;

        // 1. Prevediamo i valori Q per lo stato successivo
        const maxNextQValue = Math.max(...nextTurnData.targetQValues);

        // 2. Calcoliamo il target Q-value per l'azione presa
        const target = reward + this.discountFactor * maxNextQValue;

        // 3. Prevediamo i valori Q per lo stato attuale (riutilizzato da calcolo precedente)

        // 4. Sostituiamo il valore Q corrispondente all'azione presa con il target calcolato
        const action: number = parseInt(Object.keys(this.movesByIndex).find(i => this.movesByIndex[parseInt(i)] === prevMove) || '');

        const targetQValues: number[] = prevTargetQValues.slice();
        targetQValues[action] = target;

        // 5. Eseguiamo un passo di addestramento per il modello
        const targetTensor = tf.tensor([targetQValues]);
        await this.model.fit(prevStateTensor, targetTensor, { epochs: 1 });

        return {
            reward: reward
        }
    }

    // TODO: Missing the final turn. Also, the turn before seems never received
    // This is because of this log here:
    // INFO 17:24:32.855131 Turn: 114, Snakes Alive: [Roger123], Food: 12, Hazards: 0
    // INFO 17:24:32.858153 Turn: 115, Snakes Alive: [], Food: 12, Hazards: 0
    // INFO 17:24:33.635361 Game completed after 116 turns.
    public async trainAll(endingState: GameState): Promise<{
        reward: number;
    }> {
        await this.play(endingState);

        let i = 0;
        let reward: number = 0;
        while (this.prevGameDatas.has(i) && this.prevGameDatas.has(i + 1)) {
            const currentTurn = this.prevGameDatas.get(i) as TurnData;
            const nextTurn = this.prevGameDatas.get(i + 1) as TurnData;

            const trainResult = await this.train(currentTurn, nextTurn);

            reward += trainResult.reward;

            i++;
        }

        this.prevGameDatas.clear();

        return {
            reward: reward
        }
    }

    public async play(gameState: GameState): Promise<{
        move: Moves;
        wasValid: boolean;
    }> {
        if (this.prevGameDatas.has(gameState.turn)) {
            throw new Error("Turn already played");
        }

        const newStateTensor: tf.Tensor = this.mapStateToInput(gameState);

        const qValues = await (this.model.predict(newStateTensor) as tf.Tensor<tf.Rank>).data();
        const moveIndex: number = qValues.indexOf(Math.max(...qValues));

        let move: Moves = this.movesByIndex[moveIndex];

        const validMoves: Record<Moves, boolean> = {
            up: true,
            down: true,
            left: true,
            right: true
        };

        // We've included code to prevent your Battlesnake from moving backwards
        const myNeck = gameState.you.body[1];

        if (myNeck.x < gameState.you.head.x) {        // Neck is left of head, don't move left
            validMoves.left = false;

        } else if (myNeck.x > gameState.you.head.x) { // Neck is right of head, don't move right
            validMoves.right = false;

        } else if (myNeck.y < gameState.you.head.y) { // Neck is below head, don't move down
            validMoves.down = false;

        } else if (myNeck.y > gameState.you.head.y) { // Neck is above head, don't move up
            validMoves.up = false;
        }

        // Prevent your Battlesnake from moving out of bounds
        if (gameState.you.head.x === 0) {
            validMoves.left = false;
        }
        if (gameState.you.head.x === gameState.board.width - 1) {
            validMoves.right = false;
        }
        // Note: y coord starts from the bottom
        if (gameState.you.head.y === 0) {
            validMoves.down = false;
        }
        if (gameState.you.head.y === gameState.board.height - 1) {
            validMoves.up = false;
        }

        const isMoveValid = validMoves[move];
        // if (!isMoveValid) {
        //     const safeMoves: Moves[] = (Object.keys(validMoves) as Moves[]).filter(move => validMoves[move]);
        //     const randomMove: Moves = safeMoves[Math.floor(Math.random() * safeMoves.length)] || Moves.up;
        //     // console.warn(`${gameState.turn}: Not valid move '${move}'. Picking '${randomMove}'`);

        //     move = randomMove;
        // }

        this.prevGameDatas.set(gameState.turn, {
            targetQValues: [...qValues],
            stateTensor: newStateTensor,
            move: move,
            isMoveValid: isMoveValid,
            health: gameState.you.health,
            turn: gameState.turn
        });

        return {
            move: move,
            wasValid: isMoveValid
        };
    }

    private mapStateToInput(state: GameState): tf.Tensor {
        const boardInput: number[] = new Array(state.board.width * state.board.height).fill(0);

        function getInputIndex(coord: Coord): number {
            // Note: y coord starts from the bottom
            return (state.board.height - coord.y - 1) * state.board.width + coord.x;
        }

        for (const food of state.board.food) {
            boardInput[getInputIndex(food)] = 1;
        }

        for (const hazard of state.board.hazards) {
            boardInput[getInputIndex(hazard)] = -1;
        }

        let myselfFound = false;
        for (const snake of state.board.snakes) {
            const isMyself: boolean = snake.id === state.you.id;
            if (isMyself) {
                myselfFound = true;
            }

            for (const body of snake.body) {
                boardInput[getInputIndex(body)] = isMyself ? 2 : -2;
            }

            boardInput[getInputIndex(snake.head)] = isMyself ? 3 : -3;
        }

        if (!myselfFound) {
            const isMyself: boolean = state.you.id === state.you.id;
            if (isMyself) {
                myselfFound = true;
            }

            for (const body of state.you.body) {
                boardInput[getInputIndex(body)] = isMyself ? 2 : -2;
            }

            boardInput[getInputIndex(state.you.head)] = isMyself ? 3 : -3;
        }

        let stringified: string = "";
        for (let i = 0; i < boardInput.length; i++) {
            if ((i % state.board.width) === 0) {
                stringified += '\n';
            }

            stringified += `${boardInput[i]}`;
        }

        // Reshape allows to set the correct dimension to the tensor
        const inputTensor = tf.tensor(boardInput).reshape([-1, this.inputShape]);
        return inputTensor;
    }
}