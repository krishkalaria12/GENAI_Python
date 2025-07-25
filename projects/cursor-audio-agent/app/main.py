import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
# custom imports
from .graph import compile_graph_with_checkpointer
from .config import DB_URI, checkpointer_mongo_config
from .graph import State

# obtain audio from the microphone
def obtain_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # noise cancellation
        r.adjust_for_ambient_noise(source=source)
        r.pause_threshold = 3
        print("Say something!")
        audio = r.listen(source)
    print("Processing audio...")
    
    try:
        sst = r.recognize_google(audio)
        print("You said: " + sst)
        return sst
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Recognition error: {e}")
        return None

def main():
    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)
        
        print("🎤 Voice Recognition Loop Started!")
        print("Commands:")
        print("  - Say 'exit', 'quit', or 'stop' to end")
        print("  - Say 'pause' to temporarily pause listening")
        print("  - Press Ctrl+C to force quit")
        print("-" * 50)
        
        paused = False
        interaction_count = 0
        
        while True:
            try:
                # Handle pause state
                if paused:
                    print("🔇 Listening paused. Say 'resume' or 'continue' to resume...")
                
                sst = obtain_audio()
                
                # Skip if no audio was recognized
                if sst is None:
                    continue
                
                # Handle pause/resume commands
                if sst.lower() in ['pause', 'stop listening']:
                    paused = True
                    print("⏸️ Paused listening. Say 'resume' to continue...")
                    continue
                elif sst.lower() in ['resume', 'continue', 'start listening'] and paused:
                    paused = False
                    print("▶️ Resumed listening...")
                    continue
                elif paused:
                    continue
                
                # Check for exit commands
                if sst.lower() in ['exit', 'quit', 'stop', 'end', 'goodbye']:
                    print("👋 Goodbye!")
                    break
                
                # Increment interaction counter
                interaction_count += 1
                
                # Process the recognized speech
                state = State(
                    messages=[
                        {
                            "role": "user",
                            "content": sst,
                        }
                    ]
                )
                
                print(f"\n🔄 Processing request #{interaction_count}...")
                for event in graph_with_mongo.stream(state, checkpointer_mongo_config, stream_mode="values"):
                    if "messages" in event:
                        event["messages"][-1].pretty_print()
                
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n⚠️ Program interrupted by user")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                print("🔄 Continuing to listen...")
                continue

if __name__ == "__main__":
    main()