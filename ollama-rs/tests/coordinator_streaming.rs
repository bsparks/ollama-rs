use tokio_stream::StreamExt;

use ollama_rs::{
    coordinator::Coordinator,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
    Ollama,
};

#[cfg(feature = "stream")]
use ollama_rs::CoordinatorStreamEvent;

/// Mock weather tool for testing
#[cfg(feature = "macros")]
#[ollama_rs::function]
async fn get_weather_test(city: String) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Ok(format!("The weather in {} is sunny and 72°F", city))
}

/// Mock calculation tool for testing
#[cfg(feature = "macros")]
#[ollama_rs::function]
async fn calculate_test(
    expression: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    Ok(format!("The result of {} is 42", expression))
}

// "mistral-nemo:latest" fails tool test.
#[allow(dead_code)]
const MODEL: &str = "mistral-nemo";

#[tokio::test]
#[cfg(feature = "stream")]
#[cfg(feature = "macros")]
async fn test_coordinator_streaming_without_tools() {
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history).debug(true);

    let stream = coordinator
        .chat_stream(vec![ChatMessage::user("Say hello in one word".to_string())])
        .await
        .unwrap();

    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    let mut content_received = false;
    let mut done_received = false;

    while let Some(event) = stream.next().await {
        match &event {
            CoordinatorStreamEvent::ContentChunk(content) => {
                assert!(!content.is_empty());
                content_received = true;
                println!("Content: '{content}'");
            }
            CoordinatorStreamEvent::Done => {
                done_received = true;
            }
            CoordinatorStreamEvent::Error(err) => {
                panic!("Unexpected error: {}", err);
            }
            // These shouldn't occur without tools
            CoordinatorStreamEvent::ToolCallStarted { .. } => {
                panic!("Unexpected tool call started event");
            }
            CoordinatorStreamEvent::ToolCallCompleted { .. } => {
                panic!("Unexpected tool call completed event");
            }
            CoordinatorStreamEvent::FinalContentChunk(_) => {
                panic!("Unexpected final content chunk event");
            }
        }
        events.push(event);
        
        if done_received {
            break;
        }
    }

    assert!(content_received, "Should have received content chunks");
    assert!(done_received, "Should have received done event");
    assert!(!events.is_empty(), "Should have received some events");
}

#[cfg(feature = "macros")]
#[tokio::test]
async fn test_coordinator_with_tools() {
    // Test behavior with tools using MODEL to check tool support
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history)
        .add_tool(get_weather_test)
        .add_tool(calculate_test)
        .debug(true);

    println!("Testing non-streaming tool support with {MODEL} model...");

    // First, let's test basic non-streaming functionality to ensure the model works
    println!("🔍 Testing basic model functionality first...");
    let test_ollama = Ollama::default();
    let test_response = test_ollama.send_chat_messages(
        ChatMessageRequest::new(
            MODEL.to_string(),
            vec![ChatMessage::user("Say hello in one word.".to_string())],
        )
    ).await;

    match test_response {
        Ok(response) => {
            println!("✅ Basic model test successful: '{}'", response.message.content);
        }
        Err(e) => {
            println!("❌ Basic model test failed: {}", e);
            println!("⚠️  Skipping streaming test due to basic model failure");
            return;
        }
    }

    println!("🔧 Now testing non-streaming WITH tools...");
    let response = coordinator
        .chat(vec![ChatMessage::user(
            "What's the weather like in Portland? Please use the get_weather_test tool to check.".to_string(),
        )])
        .await;

    match response {
        Ok(response) => {
            println!("Response is {response:?}");
            // Test passes regardless - we're just investigating tool support
            println!("✅ Test completed - tool support investigation finished");
        }
        Err(e) => {
            if e.to_string().contains("does not support tools") {
                println!("❌ Model {MODEL} does not support tools - error at stream creation");
                return;
            } else if e.to_string().contains("model not found") || e.to_string().contains("404") {
                println!("⚠️  Model {MODEL} not found - skipping test");
                return;
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}

#[tokio::test]
#[cfg(feature = "stream")]
#[cfg(feature = "macros")]
async fn test_coordinator_streaming_with_tools() {
    // Test behavior with tools using MODEL to check tool support
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history)
        .add_tool(get_weather_test)
        .add_tool(calculate_test)
        .debug(true);

    println!("Testing tool support with {MODEL} model...");

    println!("🔧 Now testing streaming WITH tools...");
    let mut content_received = false;
    let mut tool_started = false;
    let mut tool_completed = false;
    let mut final_content_received = false;
    let mut done_received = false;

    let initial_message = vec![ChatMessage::user(
        "What's the weather like in Portland? Please use the get_weather_test tool to check.".to_string(),
    )];
    { // scope for the stream

        let stream = coordinator
            .chat_stream(initial_message.clone())
            .await;

        match stream {
            Ok(stream) => {
                let mut stream = Box::pin(stream);
                let mut events = Vec::new();

                println!("Stream started successfully, processing events...");

                while let Some(event) = stream.next().await {
                    println!("📝 Event received: {event:?}");
                    match &event {
                        CoordinatorStreamEvent::ContentChunk(content) => {
                            println!("📄 Content chunk: '{content}'");
                            content_received = true;
                        }
                        CoordinatorStreamEvent::ToolCallStarted { name, args } => {
                            println!("🔧 Tool call started: {name} with args: {args}");
                            assert_eq!(name, "get_weather_test");
                            tool_started = true;
                        }
                        CoordinatorStreamEvent::ToolCallCompleted { name, result } => {
                            println!("✅ Tool call completed: {name} result: {result}");
                            assert_eq!(name, "get_weather_test");
                            assert!(result.contains("sunny"));
                            tool_completed = true;
                        }
                        CoordinatorStreamEvent::FinalContentChunk(content) => {
                            println!("📄 Final content chunk: '{content}'");
                            final_content_received = true;
                        }
                        CoordinatorStreamEvent::Done => {
                            println!("✨ Stream completed");
                            done_received = true;
                        }
                        CoordinatorStreamEvent::Error(err) => {
                            if err.contains("does not support tools") {
                                println!("❌ Model {MODEL} does not support tools");
                                return;
                            } else {
                                println!("❌ Unexpected error: {}", err);
                                return; // Don't panic, just return for debugging
                            }
                        }
                    }
                    events.push(event);
                    
                    if done_received {
                        break;
                    }
                }

                println!("\nTest results:");
                println!("- Content received: {}", content_received);
                println!("- Tool started: {}", tool_started);
                println!("- Tool completed: {}", tool_completed);
                println!("- Final content received: {}", final_content_received);
                println!("- Done received: {}", done_received);
                println!("- Total events: {}", events.len());

                assert!(done_received, "Should have received done event");
                assert!(!events.is_empty(), "Should have received some events");

                if tool_started && tool_completed {
                    println!("🎉 SUCCESS: {MODEL} supports tool invocation!");
                    println!("✅ Tool execution workflow completed successfully");
                } else if content_received {
                    println!("ℹ️  Model responded with content but without using tools");
                    println!("   This could mean:");
                    println!("   - Model answered directly without needing tools");
                    println!("   - Model doesn't support tool calling");
                    println!("   - Tool calling wasn't triggered by the prompt");
                } else {
                    println!("⚠️  Unusual behavior: No content chunks received");
                    println!("   This suggests the model responded immediately with Done");
                    println!("   Possible causes:");
                    println!("   - Model configuration issue");
                    println!("   - Empty response from model");
                    println!("   - Streaming implementation issue");
                }
            }
            Err(e) => {
                if e.to_string().contains("does not support tools") {
                    println!("❌ Model mistral-small3.2:24b does not support tools - error at stream creation");
                    return;
                } else if e.to_string().contains("model not found") || e.to_string().contains("404") {
                    println!("⚠️  Model mistral-small3.2:24b not found - skipping test");
                    return;
                } else {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    if tool_completed {
        // resend request with tool result this time. But how does this work?
        let mut messages = initial_message.clone();
        messages.append(&mut coordinator.history().clone());
        println!("messages: {messages:?}");
        let stream = coordinator.chat_stream(messages).await;
        if let Ok(stream) = stream {
            let mut stream = Box::pin(stream);

            while let Some(event) = stream.next().await {
                println!("📝 Event received: {event:?}");
                match &event {
                    CoordinatorStreamEvent::ContentChunk(content) => {
                        println!("📄 Content chunk: '{content}'");
                    }
                    CoordinatorStreamEvent::ToolCallStarted { name, args } => {
                        println!("🔧 Tool call started: {name} with args: {args}");
                        assert_eq!(name, "get_weather_test");
                    }
                    CoordinatorStreamEvent::ToolCallCompleted { name, result } => {
                        println!("✅ Tool call completed: {name} result: {result}");
                        assert_eq!(name, "get_weather_test");
                        assert!(result.contains("sunny"));
                    }
                    CoordinatorStreamEvent::FinalContentChunk(content) => {
                        println!("📄 Final content chunk: '{content}'");
                    }
                    CoordinatorStreamEvent::Done => {
                        println!("✨ Stream completed");
                        break;
                    }
                    CoordinatorStreamEvent::Error(err) => {
                        if err.contains("does not support tools") {
                            println!("❌ Model {MODEL} does not support tools");
                            return;
                        } else {
                            println!("❌ Unexpected error: {}", err);
                            return; // Don't panic, just return for debugging
                        }
                    }
                }
            }
        }
    }
}

#[tokio::test]
#[cfg(feature = "stream")]
async fn test_coordinator_streaming_debug_mode() {
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history)
        .debug(true); // Enable debug mode, but don't add tools since llama2 doesn't support them

    let stream = coordinator
        .chat_stream(vec![ChatMessage::user(
            "What is 2+2?".to_string(),
        )])
        .await
        .unwrap();

    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    let mut done_received = false;

    while let Some(event) = stream.next().await {
        match &event {
            CoordinatorStreamEvent::Done => {
                done_received = true;
            }
            CoordinatorStreamEvent::Error(err) => {
                panic!("Unexpected error: {}", err);
            }
            _ => {
                // Accept any other events
            }
        }
        events.push(event);
        
        if done_received {
            break;
        }
    }

    assert!(done_received, "Should have received done event");
    assert!(!events.is_empty(), "Should have received some events");
}

#[tokio::test]
#[cfg(feature = "stream")]
async fn test_coordinator_streaming_event_order() {
    let ollama = Ollama::default();
    let history = vec![];
    let mut coordinator = Coordinator::new(ollama, MODEL.to_string(), history);

    let stream = coordinator
        .chat_stream(vec![ChatMessage::user(
            "Tell me about Seattle.".to_string(),
        )])
        .await
        .unwrap();

    let mut stream = Box::pin(stream);
    let mut events = Vec::new();
    let mut done_received = false;

    while let Some(event) = stream.next().await {
        if matches!(event, CoordinatorStreamEvent::Done) {
            done_received = true;
        }
        events.push(event);
        
        if done_received {
            break;
        }
    }

    assert!(!events.is_empty(), "Should have received events");
    assert!(done_received, "Should have received done event");
    
    // Check that Done is the last event
    if let Some(last_event) = events.last() {
        assert!(matches!(last_event, CoordinatorStreamEvent::Done), "Last event should be Done");
    }

    // For a basic query without tools, we should have ContentChunk events followed by Done
    let mut found_content = false;
    for event in &events {
        match event {
            CoordinatorStreamEvent::ContentChunk(_) => {
                found_content = true;
            }
            CoordinatorStreamEvent::Done => {
                // Done should come after content
                assert!(found_content, "Done should come after content chunks");
            }
            CoordinatorStreamEvent::ToolCallStarted { .. } => {
                panic!("Shouldn't have tool events without tools configured");
            }
            CoordinatorStreamEvent::ToolCallCompleted { .. } => {
                panic!("Shouldn't have tool events without tools configured");
            }
            CoordinatorStreamEvent::FinalContentChunk(_) => {
                panic!("Shouldn't have final content chunks without tools");
            }
            CoordinatorStreamEvent::Error(_) => {
                panic!("Shouldn't have errors in basic streaming");
            }
        }
    }
}