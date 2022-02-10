use std::{
    sync::mpsc,
    thread::{self, JoinHandle},
};

use anyhow::Result;
use rust_bert::pipelines::sentiment::{Sentiment, SentimentConfig, SentimentModel};
use tokio::{sync::oneshot, task};

#[tokio::main]
async fn main() -> Result<()> {
    let (_handle, classifier) = SentimentClassifier::spawn();

    let texts = vec![
        "Classify this positive text".to_owned(),
        "Classify this negative text".to_owned(),
    ];
    let sentiments = classifier.predict(texts).await?;
    println!("Results: {:?}", sentiments);

    Ok(())
}

/// Message type for internal channel, passing around texts and return value
/// senders
type Message = (Vec<String>, oneshot::Sender<Vec<Sentiment>>);

/// Runner for sentiment classification
#[derive(Debug, Clone)]
pub struct SentimentClassifier {
    sender: mpsc::SyncSender<Message>,
}

impl SentimentClassifier {
    /// Spawn a classifier on a separate thread and return a classifier instance
    /// to interact with it
    pub fn spawn() -> (JoinHandle<Result<()>>, SentimentClassifier) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(receiver));
        (handle, SentimentClassifier { sender })
    }

    /// The classification runner itself
    fn runner(receiver: mpsc::Receiver<Message>) -> Result<()> {
        // Needs to be in sync runtime, async doesn't work
        let model = SentimentModel::new(SentimentConfig::default())?;

        while let Ok((texts, sender)) = receiver.recv() {
            let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
            let sentiments = model.predict(texts);
            sender.send(sentiments).expect("sending results");
        }

        Ok(())
    }

    /// Make the runner predict a sample and return the result
    pub async fn predict(&self, texts: Vec<String>) -> Result<Vec<Sentiment>> {
        let (sender, receiver) = oneshot::channel();
        task::block_in_place(|| self.sender.send((texts, sender)))?;
        Ok(receiver.await?)
    }
}
