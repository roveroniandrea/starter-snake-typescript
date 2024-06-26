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

import { writeFile } from 'fs/promises';
import runServer from './server';
import { setIsTrainingInProgress } from './shared';
import { SnakeAgent } from './snakeAgent';
import { GameState, InfoResponse, MoveResponse } from './types';

async function main(): Promise<void> {

  const snakeAgent = await SnakeAgent.load("models/agent-004", true);

  let timeStats: {
    gameIndex: number;
    trainingTimeMs: number;
    playingTimeMs: number;
    validMoves: number;
  } = {
    gameIndex: 0,
    trainingTimeMs: 0,
    playingTimeMs: 0,
    validMoves: 0
  }


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
    // console.log("GAME START");
  }

  // end is called when your Battlesnake finishes a game
  async function end(gameState: GameState): Promise<void> {
    setIsTrainingInProgress(true);
    const now = new Date().getTime();
    const trainResult = await snakeAgent.trainAll(gameState);
    const finalTrainingTime = new Date().getTime() - now;

    console.log(`
      GAME #${timeStats.gameIndex} ENDED:
      Turns ${gameState.turn}
      Avg training time during game: ${timeStats.trainingTimeMs / gameState.turn}ms
      Avg playing time: ${timeStats.playingTimeMs / gameState.turn}ms
      Final training time: ${finalTrainingTime}ms
      Total valid moves: ${timeStats.validMoves} / ${gameState.turn - 1} (${((timeStats.validMoves * 100) / (gameState.turn - 1)).toFixed(2)})%
      Train reward: ${trainResult.reward}
      `);

      await snakeAgent.save("models/agent-004");
      await writeFile('models/agent-004/scores.csv', `${gameState.turn},${trainResult.reward}\n`, { flag: 'a' });

    timeStats = {
      gameIndex: timeStats.gameIndex + 1,
      trainingTimeMs: 0,
      playingTimeMs: 0,
      validMoves: 0
    };
    setIsTrainingInProgress(false);
  }

  // move is called on every turn and returns your next move
  // Valid moves are "up", "down", "left", or "right"
  // See https://docs.battlesnake.com/api/example-move for available data
  async function move(gameState: GameState): Promise<MoveResponse> {
    // if (gameState.turn && false) {
    //   const now = new Date().getTime();
    //   await snakeAgent.train(gameState);

    //   timeStats.trainingTimeMs += (new Date().getTime()) - now;
    // }

    const now = new Date().getTime();
    const {
      move,
      wasValid
    } = await snakeAgent.play(gameState);
    timeStats.playingTimeMs += (new Date().getTime()) - now;

    if (wasValid) {
      timeStats.validMoves++;
    }

    return { move: move };
  }

  runServer({
    info: info,
    start: start,
    move: move,
    end: end
  });

}

main();