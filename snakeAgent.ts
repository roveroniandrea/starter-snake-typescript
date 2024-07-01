import * as tf from '@tensorflow/tfjs-node';
import { access, mkdir } from 'fs/promises';
import { Coord, GameState } from './types';
import { Moves } from './utils';
import { boardInputToLocalSpace, isCollisionWithOthersLost, isCollisionWithSelf, isOutsideBounds, isStarved, moveToWorldSpace, printBoard } from './utils/gameUtils';

type TurnData = {
    targetQValues: number[];
    /** Rotated board passed as input */
    stateTensor: tf.Tensor;
    /** Move oriented relative to the snake's space */
    localSpaceMove: Moves;
    /** Move oriented relative to the board's space */
    worldSpaceMove: Moves;
    /** Total number of equal moves in the previous turns (0-based) */
    equalMovesCount: number;
    gameState: GameState;
};

export class SnakeAgent {

    private readonly model: tf.Sequential;
    private readonly discountFactor: number = 0.8;
    private readonly learningRate: number = 0.2;
    private readonly movesByIndex: Moves[] = [
        Moves.up,
        Moves.down,
        Moves.left,
        Moves.right,
    ];
    private readonly inputShape: number = (6 + 2) * (6 + 2) + 1;
    private readonly epsilon: number = 0.01;
    private prevGameDatas: Map<number, TurnData> = new Map();

    constructor(model?: tf.Sequential) {
        this.model = model || this.createModel();
    }

    private createModel(): tf.Sequential {
        // The number of hidden neurons should be between the size of the input layer and the size of the output layer
        // The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer
        // The number of hidden neurons should be less than twice the size of the input layer

        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 14, inputShape: [this.inputShape], activation: 'relu', useBias: true }));
        model.add(tf.layers.dense({ units: 14, activation: 'relu', useBias: true }));
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
            localSpaceMove: prevLocalSpaceMove,
            gameState: prevGameState
        } = turnData;

        const {
            gameState: nextGameState
        } = nextTurnData;

        const reward: number = (() => {
            if (isOutsideBounds(nextGameState.board, nextGameState.you)) {
                // Means that I'm outside the game board
                return -1;
            }
            if (isStarved(nextGameState.you)) {
                // Means I should have eaten more
                return -1;
            }
            if (isCollisionWithSelf(nextGameState.you)) {
                // Means that I've collided with myself
                return -1;
            }
            if (isCollisionWithOthersLost(nextGameState.you, nextGameState.board.snakes)) {
                // Means that I've collided with someone else and, in case of head-to-head collision, I've lost
                return -1;
            }
            if (nextTurnData.equalMovesCount > 2 && [Moves.right, Moves.left].includes(prevLocalSpaceMove)) {
                // Penalty for turning in the same direction more than three times
                return -0.5;
            }
            // TODO: Is on hazard

            if (!nextGameState.board.snakes.some(sn => sn.id === nextGameState.you.id)) {
                // Means that I've been eliminated
                // This is a generic "lose" check, more specific ones are before
                return -1;
            }

            if (nextGameState.you.health >= prevGameState.you.health) {
                // I've eaten a fruit (including when already at full health)
                return 1;
            }

            // Otherwise, a turn is passed with nothing special
            return 0.1;
        })();


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
        // 2. Calcoliamo il target Q-value per l'azione presa
        // 3. Prevediamo i valori Q per lo stato attuale (riutilizzato da calcolo precedente)
        // 4. Sostituiamo il valore Q corrispondente all'azione presa con il target calcolato

        const actionIndex: number = parseInt(Object.keys(this.movesByIndex).find(i => this.movesByIndex[parseInt(i)] === prevLocalSpaceMove) || '');

        // Bellman equation
        const maxNextQValue = Math.max(...nextTurnData.targetQValues);

        const target = prevTargetQValues[actionIndex] + this.learningRate * (reward + this.discountFactor * maxNextQValue - prevTargetQValues[actionIndex]);

        const targetQValues: number[] = prevTargetQValues.slice();
        targetQValues[actionIndex] = target;

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

        const validMovesInWorldSpace: Record<Moves, boolean> = {
            [Moves.up]: true,
            [Moves.down]: true,
            [Moves.left]: true,
            [Moves.right]: true
        };

        // We've included code to prevent your Battlesnake from moving backwards
        const myNeck = gameState.you.body[1];

        let heading: Moves | null = Moves.up;

        if (myNeck.x < gameState.you.head.x) {        // Neck is left of head, don't move left
            validMovesInWorldSpace.left = false;
            heading = Moves.right;

        } else if (myNeck.x > gameState.you.head.x) { // Neck is right of head, don't move right
            validMovesInWorldSpace.right = false;
            heading = Moves.left;

        } else if (myNeck.y < gameState.you.head.y) { // Neck is below head, don't move down
            validMovesInWorldSpace.down = false;
            heading = Moves.up;

        } else if (myNeck.y > gameState.you.head.y) { // Neck is above head, don't move up
            validMovesInWorldSpace.up = false;
            heading = Moves.down;
        }

        // Prevent your Battlesnake from moving out of bounds
        if (gameState.you.head.x === 0) {
            validMovesInWorldSpace.left = false;
        }
        if (gameState.you.head.x === gameState.board.width - 1) {
            validMovesInWorldSpace.right = false;
        }
        // Note: y coord starts from the bottom
        if (gameState.you.head.y === 0) {
            validMovesInWorldSpace.down = false;
        }
        if (gameState.you.head.y === gameState.board.height - 1) {
            validMovesInWorldSpace.up = false;
        }

        // TODO: Also check for collision with own body (and maybe other snakes) to see if the move is invalid?

        const newStateTensor: tf.Tensor = this.mapStateToInput(gameState, heading);

        const qValues = await (this.model.predict(newStateTensor) as tf.Tensor<tf.Rank>).data();
        const moveIndex: number = qValues.indexOf(Math.max(...qValues));

        // Chose a random move based on an epsilon value
        // The move is returned as local space, but a check is done in order to always return a valid move if possible
        const chooseRandom: () => Moves = () => {
            const remainingMoves: Moves[] = this.movesByIndex.filter(move => validMovesInWorldSpace[move]);

            if (remainingMoves.length) {
                return remainingMoves[Math.floor(Math.random() * remainingMoves.length)];
            }

            return this.movesByIndex[Math.floor(Math.random() * this.movesByIndex.length)];
        }

        const localSpaceMove: Moves = Math.random() < this.epsilon ?
            chooseRandom()
            : this.movesByIndex[moveIndex];

        // The move needs to be converted to world space.
        const worldSpaceMove = moveToWorldSpace(localSpaceMove, heading);
        // To check if the move is valid, use the rotated one
        const isMoveValid = validMovesInWorldSpace[worldSpaceMove];

        const prevTurnData: TurnData | null = this.prevGameDatas.get(gameState.turn - 1) || null;

        this.prevGameDatas.set(gameState.turn, {
            targetQValues: [...qValues],
            stateTensor: newStateTensor,
            localSpaceMove: localSpaceMove,
            worldSpaceMove: worldSpaceMove,
            // Counting how many times the move was the same
            equalMovesCount: prevTurnData?.localSpaceMove === localSpaceMove ? prevTurnData.equalMovesCount + 1 : 0,
            gameState: gameState
        });


        return {
            move: worldSpaceMove,
            wasValid: isMoveValid
        };
    }

    private mapStateToInput(state: GameState, heading: Moves): tf.Tensor {
        const worldSpaceBoard: number[][] = new Array(state.board.height + 2)
            .fill(null)
            .map(() => new Array(state.board.width + 2).fill(0));

        // The border of the board shoul be initialized to -1
        for (let x = -1; x < worldSpaceBoard[0].length - 1; x++) {
            setBoardValue({ x: x, y: -1 }, -1);
            setBoardValue({ x: x, y: state.board.height }, -1);
        }
        for (let y = -1; y < worldSpaceBoard.length - 1; y++) {
            setBoardValue({ x: -1, y: y }, -1);
            setBoardValue({ x: state.board.width, y: y }, -1);
        }


        function setBoardValue(coord: Coord, value: number): void {
            worldSpaceBoard[coord.y + 1][coord.x + 1] = value;
        }

        for (const food of state.board.food) {
            setBoardValue(food, 1);
        }

        for (const hazard of state.board.hazards) {
            setBoardValue(hazard, -1);
        }

        let myselfFound = false;
        for (const snake of state.board.snakes) {
            const isMyself: boolean = snake.id === state.you.id;
            if (isMyself) {
                myselfFound = true;
            }

            for (const body of snake.body) {
                setBoardValue(body, isMyself ? 2 : -2);
            }

            setBoardValue(snake.head, isMyself ? 3 : -3)
        }

        if (!myselfFound) {
            for (const body of state.you.body) {
                setBoardValue(body, 2);
            }

            setBoardValue(state.you.head, 3);
        }

        const localSpaceBoard: tf.Tensor = boardInputToLocalSpace(tf.tensor2d(worldSpaceBoard, [state.board.width + 2, state.board.height + 2]), heading);

        // console.log(JSON.stringify(worldSpaceBoard));
        // printBoard(localSpaceBoard);
        const healthRatio = Math.max(Math.min(state.you.health / 100, 1), 0);

        // Reshape allows to set the correct dimension to the tensor
        const inputTensor = tf.concat([
            localSpaceBoard.reshape([-1, localSpaceBoard.size]),
            tf.tensor2d([[healthRatio]], [1, 1])
        ], 1);

        return inputTensor;
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