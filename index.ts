// Welcome to
// __________         __    __  .__                               __
// \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
//  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
//  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
//  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
//
// This file can be a nice home for your Battlesnake logic and helper functions.
//
// To get you started we've included code to prevent your Battlesnake from moving backwards.
// For more info see docs.battlesnake.com

import runServer from './server';
import { SnakeAgent } from './snakeAgent';
import { Moves } from './utils';
import { GameState, InfoResponse, MoveResponse } from './types';

const snakeAgent = new SnakeAgent();



let prevMove: Moves | null = null;

// info is called when you create your Battlesnake on play.battlesnake.com
// and controls your Battlesnake's appearance
// TIP: If you open your Battlesnake URL in a browser you should see this data
function info(): InfoResponse {
  console.log("INFO");

  return {
    apiversion: "1",
    author: "",       // TODO: Your Battlesnake Username
    color: "#888888", // TODO: Choose color
    head: "default",  // TODO: Choose head
    tail: "default",  // TODO: Choose tail
  };
}

// start is called when your Battlesnake begins a game
function start(gameState: GameState): void {
  console.log("GAME START");
}

// end is called when your Battlesnake finishes a game
function end(gameState: GameState): void {
  if (prevMove) {
    snakeAgent.train(prevMove, gameState);
  }
  console.log("GAME OVER\n");
}

// move is called on every turn and returns your next move
// Valid moves are "up", "down", "left", or "right"
// See https://docs.battlesnake.com/api/example-move for available data
function move(gameState: GameState): MoveResponse {
  if (prevMove) {
    snakeAgent.train(prevMove, gameState);
  }

  let isMoveValid: Record<Moves, boolean> = {
    up: true,
    down: true,
    left: true,
    right: true
  };

  // We've included code to prevent your Battlesnake from moving backwards
  const myHead = gameState.you.body[0];
  const myNeck = gameState.you.body[1];

  if (myNeck.x < myHead.x) {        // Neck is left of head, don't move left
    isMoveValid.left = false;

  } else if (myNeck.x > myHead.x) { // Neck is right of head, don't move right
    isMoveValid.right = false;

  } else if (myNeck.y < myHead.y) { // Neck is below head, don't move down
    isMoveValid.down = false;

  } else if (myNeck.y > myHead.y) { // Neck is above head, don't move up
    isMoveValid.up = false;
  }

  // TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
  // boardWidth = gameState.board.width;
  // boardHeight = gameState.board.height;

  let nextMove: Moves = snakeAgent.play(gameState);

  if (!isMoveValid[nextMove]) {
    const safeMoves: Moves[] = (Object.keys(isMoveValid) as Moves[]).filter(move => isMoveValid[move]);
    const randomMove: Moves = safeMoves[Math.floor(Math.random() * safeMoves.length)] || Moves.up;
    console.warn(`${gameState.turn}: Not valid move '${nextMove}. Picking '${randomMove}'`);

    nextMove = randomMove;
  }

  return { move: nextMove };
}

runServer({
  info: info,
  start: start,
  move: move,
  end: end
});
