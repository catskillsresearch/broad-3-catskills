// Add image modal functionality
document.addEventListener('DOMContentLoaded', function() {
    // Create modal elements
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close-modal">&times;</span>
            <img class="modal-image" src="" alt="Enlarged image">
        </div>
    `;
    document.body.appendChild(modal);

    // Get modal elements
    const modalImg = document.querySelector('.modal-image');
    const closeModal = document.querySelector('.close-modal');

    // Add click event to all mermaid diagrams and images
    document.querySelectorAll('.mermaid-container, .diagram-container, img.enlargeable').forEach(container => {
        container.style.cursor = 'pointer';
        container.title = 'Click to enlarge';
        container.addEventListener('click', function() {
            // For mermaid diagrams, get the SVG
            if (this.classList.contains('mermaid-container') || this.classList.contains('diagram-container')) {
                const svg = this.querySelector('svg');
                if (svg) {
                    // Create a data URL from the SVG
                    const svgData = new XMLSerializer().serializeToString(svg);
                    const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
                    const svgUrl = URL.createObjectURL(svgBlob);
                    
                    // Show the modal with the SVG
                    modal.style.display = 'block';
                    modalImg.src = svgUrl;
                }
            } else {
                // For regular images
                modal.style.display = 'block';
                modalImg.src = this.src;
            }
        });
    });

    // Close modal when clicking the close button
    closeModal.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // Close modal when clicking outside the image
    modal.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Add click-to-copy functionality for code blocks
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.copy-button').forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.parentElement.nextElementSibling.querySelector('code');
            const textToCopy = codeBlock.textContent;
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change button text temporarily
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });
});

// Toggle mobile menu
document.addEventListener('DOMContentLoaded', function() {
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    const mainNav = document.querySelector('.main-nav');
    
    if (menuToggle && mainNav) {
        menuToggle.addEventListener('click', function() {
            mainNav.classList.toggle('active');
            this.classList.toggle('active');
        });
    }
});

// Toggle accordion items
document.addEventListener('DOMContentLoaded', function() {
    const accordionItems = document.querySelectorAll('.accordion-item');
    
    accordionItems.forEach(item => {
        const header = item.querySelector('.accordion-header');
        const content = item.querySelector('.accordion-content');
        
        if (header && content) {
            header.addEventListener('click', function() {
                this.classList.toggle('active');
                
                if (this.classList.contains('active')) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                } else {
                    content.style.maxHeight = '0';
                }
            });
        }
    });
});
