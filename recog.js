const { tensor } = require('@tensorflow/tfjs');


theAi();

async function theAi() {

    const tf = require('@tensorflow/tfjs'),
      mobilenet = require('@tensorflow-models/mobilenet'),
      tfnode = require('@tensorflow/tfjs-node'),
      fs = require('fs-extra');

    const imageBuffer = await fs.readFile("./doggy.jpg"),
      tfimage = tfnode.node.decodeImage(imageBuffer),
      mobilenetModel = await mobilenet.load();  

    const results = await mobilenetModel.classify(tfimage);
    const maxClass = results.reduce(function(prev, current) {
      return (prev.probability > current.probability) ? prev : current;
    })["className"];
    console.log(maxClass);
  
  };
