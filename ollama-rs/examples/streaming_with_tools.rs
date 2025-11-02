use std::io::{stdout, Write};
use tokio_stream::StreamExt;

use ollama_rs::{
    coordinator::Coordinator,
    generation::chat::ChatMessage,
    Ollama,
};

#[cfg(feature = "stream")]
use ollama_rs::CoordinatorStreamEvent;

/// Get the weather for a given city (mock implementation)
#[cfg(feature = "macros")]
#[ollama_rs::function]
async fn get_weather(city: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Mock weather data - in reality this would call a weather API
    Ok(format!("The weather in {} is sunny with a temperature of 72°F", city))
}

/// Calculate distance between two cities (mock implementation)
#[cfg(feature = "macros")]
#[ollama_rs::function]
async fn calculate_distance(
    from_city: String,
    to_city: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Mock distance calculation - in reality this would use geocoding APIs
    Ok(format!("The distance from {} to {} is approximately 250 miles", from_city, to_city))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() > 2 || (args.get(1).is_some() && args[1] != "-d") {
        eprintln!("Usage: {} [-d] (to enable debugging)", args[0]);
        return Ok(());
    }

    let debug = args.get(1).is_some();

    println!("🚀 Starting streaming chat with tools example");
    println!("This example demonstrates hybrid streaming with tool notifications");
    println!("Available tools: get_weather, calculate_distance");
    println!();

    let ollama = Ollama::default();
    let history = vec![];

    let mut coordinator = Coordinator::new(ollama, "mistral-nemo:latest".to_string(), history)
        .add_tool(get_weather)
        .add_tool(calculate_distance)
        .debug(debug);

    // Test queries that should trigger tools
    let test_queries = vec![
        "What's the weather like in Portland, Oregon?",
        "How far is it from San Francisco to Los Angeles?",
        "What's a good place to visit in Seattle?", // This might not trigger tools
    ];

    for (i, query) in test_queries.iter().enumerate() {
        println!("📝 Query {}: {}", i + 1, query);
        println!("🤖 Assistant: ");

        let stream = coordinator
            .chat_stream(vec![ChatMessage::user(query.to_string())])
            .await?;

        let mut stream = Box::pin(stream);
        let mut stdout = stdout();
        while let Some(event) = stream.next().await {
            match event {
                CoordinatorStreamEvent::ContentChunk(content) => {
                    print!("{}", content);
                    stdout.flush()?;
                }
                CoordinatorStreamEvent::ToolCallStarted { name, args } => {
                    println!("\n🔧 [Calling tool: {} with args: {}]", name, args);
                }
                CoordinatorStreamEvent::ToolCallCompleted { name, result } => {
                    println!("✅ [Tool {} completed: {}]", name, result);
                    print!("🤖 Assistant: ");
                }
                CoordinatorStreamEvent::FinalContentChunk(content) => {
                    print!("{}", content);
                    stdout.flush()?;
                }
                CoordinatorStreamEvent::Done => {
                    println!("\n✨ [Conversation complete]");
                    break;
                }
                CoordinatorStreamEvent::Error(err) => {
                    eprintln!("\n❌ [Error: {}]", err);
                    break;
                }
            }
        }

        println!("\n{}", "─".repeat(80));
    }

    println!("🎉 Example completed successfully!");
    Ok(())
}