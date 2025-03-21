function toggleChat() {
    var chatWindow = document.getElementById("chat-window");
    var mainContainer = document.querySelector(".main-container");
    var chatbotBtn = document.getElementById("chatbot-btn");

    if (chatWindow.style.display === "none" || chatWindow.style.display === "") {
        chatWindow.style.display = "flex";
        chatbotBtn.innerHTML = "✉"; 
        mainContainer.classList.add("shift-left");
    } else {
        chatWindow.style.display = "none";
        chatbotBtn.innerHTML = "💬 Ask me about Shruthin"; 
        mainContainer.classList.remove("shift-left");
    }

    if (chatWindow.style.display === "flex") {
        var chatMessages = document.getElementById("chat-messages");
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function sendMessage(userInput = null) {
    var inputField = document.getElementById("user-input");
    var chatMessages = document.getElementById("chat-messages");

    if (!userInput) {
        userInput = inputField.value.trim();
    }
    
    if (userInput === "") return; 

    var userMessageElement = document.createElement("div");
    userMessageElement.classList.add("message", "user-message");
    
    var messageContent = document.createElement("div");
    messageContent.classList.add("message-content");
    messageContent.textContent = userInput;
    
    userMessageElement.appendChild(messageContent);
    chatMessages.appendChild(userMessageElement);

    var botTypingElement = document.createElement("div");
    botTypingElement.classList.add("message", "bot-message", "typing-effect");
    
    var botAvatar = document.createElement("img");
    botAvatar.src = document.querySelector(".chat-avatar").src;
    botAvatar.classList.add("message-avatar");
    botTypingElement.appendChild(botAvatar);
    
    var typingContent = document.createElement("div");
    typingContent.classList.add("message-content");
    typingContent.textContent = "Typing...";
    botTypingElement.appendChild(typingContent);
    
    chatMessages.appendChild(botTypingElement);

    chatMessages.scrollTop = chatMessages.scrollHeight;

    setTimeout(() => {
        fetch("http://127.0.0.1:5000/chat", { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userInput }),
        })
        .then(response => response.json())
        .then(data => {
            chatMessages.removeChild(botTypingElement);
            
            var botMessageElement = document.createElement("div");
            botMessageElement.classList.add("message", "bot-message");
            
            var botAvatar = document.createElement("img");
            botAvatar.src = document.querySelector(".chat-avatar").src;
            botAvatar.classList.add("message-avatar");
            botMessageElement.appendChild(botAvatar);

            var messageContent = document.createElement("div");
            messageContent.classList.add("message-content");
            
            let processedReply = processMessageContent(data.reply);
            messageContent.innerHTML = processedReply;
            
            botMessageElement.appendChild(messageContent);
            chatMessages.appendChild(botMessageElement);

            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            console.error("Error:", error);
            
            chatMessages.removeChild(botTypingElement);
            
            var errorMessageElement = document.createElement("div");
            errorMessageElement.classList.add("message", "bot-message");

            var botAvatar = document.createElement("img");
            botAvatar.src = document.querySelector(".chat-avatar").src;
            botAvatar.classList.add("message-avatar");
            errorMessageElement.appendChild(botAvatar);
            
            var messageContent = document.createElement("div");
            messageContent.classList.add("message-content");
            messageContent.textContent = "Sorry, I couldn't process your request. Please try again later.";
            errorMessageElement.appendChild(messageContent);
            
            chatMessages.appendChild(errorMessageElement);
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }, 1500); 

    inputField.value = "";
}

function processMessageContent(content) {
    let result = content;
    
    const urlMatches = content.match(/(https?:\/\/[^\s<>"']+)/g);
    
    if (urlMatches) {
        urlMatches.forEach(url => {
            if (!content.includes(`href="${url}"`) && !content.includes(`href='${url}'`)) {
                result = result.replace(
                    new RegExp(url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'),
                    `<a href="${url}" target="_blank" class="chat-link">${url}</a>`
                );
            }
        });
    }
    
    if (!result.includes("<ul>") && !result.includes("<ol>") && result.includes("\n• ")) {
        let lines = result.split("\n");
        let inList = false;
        let processedContent = "";

        for (let line of lines) {
            if (line.trim().startsWith("• ")) {
                if (!inList) {
                    processedContent += "<ul>";
                    inList = true;
                }
                processedContent += "<li>" + line.trim().substring(2) + "</li>";
            } else {
                if (inList) {
                    processedContent += "</ul>";
                    inList = false;
                }
                processedContent += "<p>" + line + "</p>";
            }
        }

        if (inList) {
            processedContent += "</ul>";
        }

        return processedContent;
    }
    
    return result;
}


function initAboutSectionEffects() {
    const aboutSection = document.getElementById('about-section');
    if (!aboutSection) return;
 
    document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;
        
        aboutSection.querySelectorAll('.skill-card').forEach((card) => {
            const offsetX = (mouseX - 0.5) * 15;
            const offsetY = (mouseY - 0.5) * 15;
            card.style.transform = `translate(${offsetX}px, ${offsetY}px) rotateX(${offsetY}deg) rotateY(${-offsetX}deg)`;
        });
        
        const paragraph = aboutSection.querySelector('p');
        if (paragraph) {
            paragraph.style.transform = `translate(${(mouseX - 0.5) * 5}px, ${(mouseY - 0.5) * 5}px)`;
        }
    });

    aboutSection.querySelectorAll('.highlight').forEach(span => {
        span.addEventListener('mouseenter', () => {
            span.style.letterSpacing = '0.5px';
            span.style.transform = 'scale(1.05)';
            span.style.display = 'inline-block';
        });
        
        span.addEventListener('mouseleave', () => {
            span.style.letterSpacing = 'normal';
            span.style.transform = 'scale(1)';
        });
    });
    
    const title = aboutSection.querySelector('h2');
    const originalText = title.textContent;
    title.textContent = '';
    
    let i = 0;
    function typeWriter() {
        if (i < originalText.length) {
            title.textContent += originalText.charAt(i);
            i++;
            setTimeout(typeWriter, 100);
        } else {
            title.classList.add('typed');
        }
    }
    

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                setTimeout(typeWriter, 500);
                observer.unobserve(entry.target);
            }
        });
    });
    
    observer.observe(title);
}

function createParticleBackground() {
    const aboutSection = document.getElementById('about-section');
    if (!aboutSection) return;
    
    const canvas = document.createElement('canvas');
    canvas.classList.add('particles-canvas');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '-1';
    canvas.style.opacity = '0.5';
    
    aboutSection.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    function resizeCanvas() {
        canvas.width = aboutSection.offsetWidth;
        canvas.height = aboutSection.offsetHeight;
    }
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const particlesArray = [];
    const numberOfParticles = 30;
    
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 5 + 1;
            this.speedX = Math.random() * 1 - 0.5;
            this.speedY = Math.random() * 1 - 0.5;
            this.color = `rgba(255, 255, 255, ${Math.random() * 0.3})`;
        }
        
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            
            if (this.size > 0.2) this.size -= 0.05;

            if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
            if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
        }
        
        draw() {
            ctx.fillStyle = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    function init() {
        for (let i = 0; i < numberOfParticles; i++) {
            particlesArray.push(new Particle());
        }
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
            particlesArray[i].draw();
        }
        requestAnimationFrame(animate);
    }
    
    init();
    animate();
}

function animateAboutText() {
    const aboutParagraph = document.querySelector('#about-section p');
    if (!aboutParagraph) return;
    
    const text = aboutParagraph.innerHTML;

    const htmlTagsAndWords = text.split(/(<.*?>)|\s+/);
    const animatedHtml = htmlTagsAndWords
        .filter(item => item)
        .map((item, index) => {
            if (item.startsWith('<') && item.endsWith('>')) {
                return item; 
            } else {
                return `<span class="word-animation" style="--word-index: ${index}">${item}</span>`;
            }
        })
        .join(' ');
    
    aboutParagraph.innerHTML = animatedHtml;
}

document.addEventListener('DOMContentLoaded', function() {

    var chatbotBtn = document.getElementById("chatbot-btn");
    chatbotBtn.innerHTML = "💬 Ask me about Shruthin";
    
    const headshotImg = document.querySelector(".headshot img");
    if (headshotImg) {
        headshotImg.style.boxShadow = "0px 20px 50px rgba(0,0,0,0.3)";
        headshotImg.style.borderRadius = "15px";
        headshotImg.style.transform = "perspective(1000px) rotateY(5deg)";
        headshotImg.style.transition = "transform 0.5s ease-in-out, box-shadow 0.5s ease-in-out";
    }
    
    document.getElementById("user-input").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    document.querySelectorAll(".query-btn").forEach(button => {
        button.addEventListener("click", function() {
            var query = this.getAttribute("data-query");
            sendMessage(query);
        });
    });

    document.getElementById('contact-button').addEventListener('click', function() {
        const contactInfo = document.getElementById('contact-info');
        contactInfo.classList.toggle('hidden');
    });

    document.getElementById('resume-button').addEventListener('click', function() {
        window.open("{{ url_for('static', filename='Resume.pdf') }}", '_blank');
    });
    
    initAboutSectionEffects();
    createParticleBackground();
    animateAboutText();
});

window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section');
    
    sections.forEach(section => {
        const sectionTop = section.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        
        if (sectionTop < windowHeight * 0.75) {
            section.classList.add('visible');
            
            if (section.id === 'about-section') {
                section.classList.add('animate');
            }
        }
    });
});

document.getElementById('resume-button').addEventListener('click', function() {
    window.open(this.getAttribute('data-resume-url'), '_blank');
});