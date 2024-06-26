import * as tf from '@tensorflow/tfjs-node';
import { access, mkdir } from 'fs/promises';
import { Coord, GameState } from './types';
import { Moves } from './utils';

type TurnData = {
    targetQValues: number[];
    /** Rotated board passed as input */
    stateTensor: tf.Tensor;
    /** Move oriented relative to the snake's space */
    localSpaceMove: Moves;
    /** Move oriented relative to the board's space */
    worldSpaceMove: Moves;
    heading: Moves;
    isMoveValid: boolean;
    health: number;
    turn: number;
    /** Total number of equal moves in the previous turns (0-based) */
    equalMovesCount: number;
};

export class SnakeAgent {

    private readonly model: tf.Sequential;
    private readonly discountFactor: number = 0.9;
    private readonly learningRate: number = 1;
    private readonly movesByIndex: Moves[] = [
        Moves.up,
        Moves.down,
        Moves.left,
        Moves.right,
    ];
    private readonly inputShape: number = 11 * 11 + 1;
    private readonly epsilon: number = 0.07;
    private prevGameDatas: Map<number, TurnData> = new Map();

    constructor(model?: tf.Sequential) {
        this.model = model || this.createModel();
    }

    private createModel(): tf.Sequential {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, inputShape: [this.inputShape], activation: 'relu', useBias: true }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu', useBias: true }));
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
            localSpaceMove: prevMove,
            isMoveValid,
            health
        } = turnData;

        const reward: number = !isMoveValid ? -1 :
            (nextTurnData.health > health) ? 1 :
                // Penalty for choosing the same move more than three times
                (nextTurnData.equalMovesCount > 2) ? -0.5 : 0;

        /* This is a trial for a custom path to follow. The agent is agent-005-custom-path
        let reward = -1;
        if ([0, 1, 2].includes(turnData.turn) && turnData.worldSpaceMove === Moves.up) {
            reward = 1;
        }
        else {
            if ([3, 4].includes(turnData.turn) && turnData.worldSpaceMove === Moves.right) {
                reward = 1;
            }
            else {
                if ([5].includes(turnData.turn) && turnData.worldSpaceMove === Moves.down) {
                    reward = 1;
                }
                else {
                    if ([6].includes(turnData.turn) && turnData.worldSpaceMove === Moves.left) {
                        reward = 1;
                    }
                    else {
                        if (turnData.turn > 6 && turnData.worldSpaceMove === Moves.left) {
                            reward = 1;
                        }
                    }
                }
            }

        }
        */
       
        // 1. Prevediamo i valori Q per lo stato successivo
        const maxNextQValue = Math.max(...nextTurnData.targetQValues);

        // 2. Calcoliamo il target Q-value per l'azione presa
        const target = this.learningRate * (reward + this.discountFactor * maxNextQValue);

        // 3. Prevediamo i valori Q per lo stato attuale (riutilizzato da calcolo precedente)

        // 4. Sostituiamo il valore Q corrispondente all'azione presa con il target calcolato
        const action: number = parseInt(Object.keys(this.movesByIndex).find(i => this.movesByIndex[parseInt(i)] === prevMove) || '');

        const targetQValues: number[] = prevTargetQValues.slice();
        targetQValues[action] = target;

        // 5. Eseguiamo un passo di addestramento per il modello
        const targetTensor = tf.tensor([targetQValues]);
        await this.model.fit(prevStateTensor, targetTensor, { epochs: 1, verbose: 0 });

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
        await this.play({
            ...endingState,
            turn: endingState.turn - 1
        });

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

        const validMoves: Record<Moves, boolean> = {
            up: true,
            down: true,
            left: true,
            right: true
        };

        // We've included code to prevent your Battlesnake from moving backwards
        const myNeck = gameState.you.body[1];

        let heading: Moves | null = Moves.up;

        if (myNeck.x < gameState.you.head.x) {        // Neck is left of head, don't move left
            validMoves.left = false;
            heading = Moves.right;

        } else if (myNeck.x > gameState.you.head.x) { // Neck is right of head, don't move right
            validMoves.right = false;
            heading = Moves.left;

        } else if (myNeck.y < gameState.you.head.y) { // Neck is below head, don't move down
            validMoves.down = false;
            heading = Moves.up;

        } else if (myNeck.y > gameState.you.head.y) { // Neck is above head, don't move up
            validMoves.up = false;
            heading = Moves.down;
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

        const newStateTensor: tf.Tensor = this.mapStateToInput(gameState, heading);

        const qValues = await (this.model.predict(newStateTensor) as tf.Tensor<tf.Rank>).data();
        const moveIndex: number = qValues.indexOf(Math.max(...qValues));

        // Chose a random move based on an epsilon value
        // The move is returned as local space, but a check is done in order to always return a valid move if possible
        const chooseRandom: () => Moves = () => {
            const remainingMoves: Moves[] = this.movesByIndex.filter(move => validMoves[move]);

            if (remainingMoves.length) {
                return remainingMoves[Math.floor(Math.random() * remainingMoves.length)];
            }

            return this.movesByIndex[Math.floor(Math.random() * this.movesByIndex.length)];
        }

        const localSpaceMove: Moves = Math.random() < this.epsilon ?
            chooseRandom()
            : this.movesByIndex[moveIndex];

        // The move needs to be converted to world space.
        const worldSpaceMove = this.moveToWorldSpace(localSpaceMove, heading);
        // To check if the move is valid, use the rotated one
        const isMoveValid = validMoves[worldSpaceMove];

        const prevTurnData: TurnData | null = this.prevGameDatas.get(gameState.turn - 1) || null;

        this.prevGameDatas.set(gameState.turn, {
            targetQValues: [...qValues],
            stateTensor: newStateTensor,
            localSpaceMove: localSpaceMove,
            worldSpaceMove: worldSpaceMove,
            heading: heading,
            isMoveValid: isMoveValid,
            health: gameState.you.health,
            turn: gameState.turn,
            // Counting how many times the move was the same
            equalMovesCount: prevTurnData?.localSpaceMove === localSpaceMove ? prevTurnData.equalMovesCount + 1 : 0
        });


        return {
            move: worldSpaceMove,
            wasValid: isMoveValid
        };
    }

    private mapStateToInput(state: GameState, heading: Moves): tf.Tensor {
        const worldSpaceBoard: number[] = new Array(state.board.width * state.board.height).fill(0);

        function clampInBoard(coord: Coord): Coord {
            return {
                x: Math.max(Math.min(coord.x, state.board.width - 1), 0),
                y: Math.max(Math.min(coord.y, state.board.height - 1), 0)
            }
        }
        function getInputIndex(coord: Coord): number {
            const clamped = clampInBoard(coord);
            // Note: y coord starts from the bottom
            return (state.board.height - clamped.y - 1) * state.board.width + clamped.x;
        }

        for (const food of state.board.food) {
            worldSpaceBoard[getInputIndex(food)] = 1;
        }

        for (const hazard of state.board.hazards) {
            worldSpaceBoard[getInputIndex(hazard)] = -1;
        }

        let myselfFound = false;
        for (const snake of state.board.snakes) {
            const isMyself: boolean = snake.id === state.you.id;
            if (isMyself) {
                myselfFound = true;
            }

            for (const body of snake.body) {
                worldSpaceBoard[getInputIndex(body)] = isMyself ? 2 : -2;
            }

            worldSpaceBoard[getInputIndex(snake.head)] = isMyself ? 3 : -3;
        }

        if (!myselfFound) {
            for (const body of state.you.body) {
                worldSpaceBoard[getInputIndex(body)] = 2;
            }

            worldSpaceBoard[getInputIndex(state.you.head)] = 3;
        }

        const localSpaceBoard: tf.Tensor = this.boardInputToLocalSpace(worldSpaceBoard, heading, state.board.width, state.board.height);

        const healthRatio = Math.max(Math.min(state.you.health / 100, 1), 0);

        // Reshape allows to set the correct dimension to the tensor
        const inputTensor = tf.concat([
            localSpaceBoard.reshape([-1, worldSpaceBoard.length]),
            tf.tensor2d([[healthRatio]], [1, 1])
        ], 1);

        return inputTensor;
    }

    private boardInputToLocalSpace(boardInput: number[], heading: Moves, width: number, height: number): tf.Tensor {
        const tensor = tf.tensor2d(boardInput, [width, height]); // TODO: Need to verify for non-squared boards

        if (heading === Moves.up) {
            return tensor;
        }

        if (heading === Moves.right) {
            // Per ruotare una matrice di 90 gradi a destra (in senso orario),
            // puoi trasporre la matrice e poi invertire l'ordine delle righe
            const transposed = tensor.transpose();
            const rotated = transposed.reverse(0);
            return rotated;
        }

        if (heading === Moves.down) {
            // Per ruotare una matrice di 180 gradi, puoi invertire sia l'ordine delle righe che delle colonne
            const rotated = tensor.reverse(0).reverse(1);
            return rotated;
        }

        if (heading === Moves.left) {
            // Per ruotare una matrice di 270 gradi a destra (in senso orario),
            // puoi trasporre la matrice e poi invertire l'ordine delle colonne
            const transposed = tensor.transpose();
            const rotated = transposed.reverse(0);
            return rotated;
        }

        throw new Error(`Invalid heading ${heading}`);
    }

    private moveToWorldSpace(move: Moves, heading: Moves): Moves {
        if (heading === Moves.up) {
            return move;
        }

        if (heading === Moves.right) {
            return move === Moves.up ? Moves.right :
                move === Moves.right ? Moves.down :
                    move === Moves.down ? Moves.left :
                        Moves.up
        }

        if (heading === Moves.down) {
            return move === Moves.up ? Moves.down :
                move === Moves.right ? Moves.left :
                    move === Moves.down ? Moves.up :
                        Moves.right
        }

        if (heading === Moves.left) {
            return move === Moves.up ? Moves.left :
                move === Moves.right ? Moves.up :
                    move === Moves.down ? Moves.right :
                        Moves.down
        }

        throw new Error(`Invalid heading ${heading}`);
    }

    public static async load(path: string, fallbackToNewModel: boolean): Promise<SnakeAgent> {
        try {
            const model = await tf.loadLayersModel(`file://${path}/model.json`);

            const sequentialModel = tf.sequential();
            model.layers.forEach(layer => {
                sequentialModel.add(layer);
            });
            sequentialModel.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

            return new SnakeAgent(sequentialModel);
        }
        catch (ex) {
            if (fallbackToNewModel) {
                return new SnakeAgent();
            }

            throw new Error(`Error  when loading model ${path}: ${(ex as any)?.message || ""}`);
        }
    }

    public async save(path: string): Promise<void> {
        try {
            await access(path);
        }
        catch (_) {
            await mkdir(path, { recursive: true });
        }
        await this.model.save(`file://${path}`);
    }
}