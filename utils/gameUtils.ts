import { Tensor } from '@tensorflow/tfjs-node';
import { Battlesnake, Board, Coord } from '../types';
import { Moves } from './utils';

export function moveToWorldSpace(move: Moves, heading: Moves): Moves {
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

export function boardInputToLocalSpace(boardInput: Tensor, heading: Moves): Tensor {
    if (heading === Moves.up) {
        return boardInput;
    }

    if (heading === Moves.right) {
        // Per ruotare una matrice di 90 gradi a destra (in senso orario),
        // puoi trasporre la matrice e poi invertire l'ordine delle righe
        const transposed = boardInput.transpose();
        const rotated = transposed.reverse(0);
        return rotated;
    }

    if (heading === Moves.down) {
        // Per ruotare una matrice di 180 gradi, puoi invertire sia l'ordine delle righe che delle colonne
        const rotated = boardInput.reverse(0).reverse(1);
        return rotated;
    }

    if (heading === Moves.left) {
        // Per ruotare una matrice di 270 gradi a destra (in senso orario),
        // puoi trasporre la matrice e poi invertire l'ordine delle colonne
        const transposed = boardInput.transpose();
        const rotated = transposed.reverse(0);
        return rotated;
    }

    throw new Error(`Invalid heading ${heading}`);
}

export function isOutsideBounds(board: Board, snake: Battlesnake): boolean {
    return snake.head.x < 0
        || snake.head.y < 0
        || snake.head.x >= board.width
        || snake.head.y >= board.height;
}

export function isStarved(snake: Battlesnake): boolean {
    return snake.health <= 0;
}

export function isCollisionWithSelf(snake: Battlesnake): boolean {
    const [head, ...body] = snake.body;

    return body.some(cell => cell.x === head.x && cell.y === head.y);
}

export function isCollisionWithOthersLost(snake: Battlesnake, others: Battlesnake[]) {
    const myHead: Coord = snake.head;

    return others.some(other => other.body.some((cell, i) => {
        // Skip if myself
        return other.id !== snake.id
            // My snake's head should be in contact with the other snake's body
            && cell.x === myHead.x
            && cell.y === myHead.y
            // And it should not be a head-to-head collision, or the other snake should be longer
            && (
                i !== 0
                || other.length >= snake.length
            )
    }))
}

export function printBoard(tensor: Tensor): void {
    tensor.print();
}