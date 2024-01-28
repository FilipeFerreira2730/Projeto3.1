
css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
         <img width="96" height="96" src="https://img.icons8.com/color/96/message-bot.png" alt="message-bot"/>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
      <img width="64" height="64" src="https://img.icons8.com/pastel-glyph/64/FFFFFF/user-male-circle.png" alt="user-male-circle"/>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
