// Script personnalisé pour Sunu Stat - ANSD
function customizeSunuStatInterface() {
    // Fonction pour personnaliser les placeholders
    function updatePlaceholders() {
        const inputElements = document.querySelectorAll('input[placeholder], textarea[placeholder]');
        inputElements.forEach(input => {
            const placeholder = input.getAttribute('placeholder');
            if (placeholder && (
                placeholder.includes('Type') || 
                placeholder.includes('message') || 
                placeholder.includes('Enter') ||
                placeholder.toLowerCase().includes('type your message') ||
                placeholder.toLowerCase().includes('écrivez votre message')
            )) {
                input.setAttribute('placeholder', 'Posez votre question sur les statistiques du Sénégal...');
            }
            
            // Pour les champs de recherche
            if (placeholder && (
                placeholder.includes('Search') || 
                placeholder.includes('search') ||
                placeholder.includes('Recherche')
            )) {
                input.setAttribute('placeholder', 'Rechercher dans les conversations...');
            }
        });
    }

    // Fonction pour supprimer complètement le watermark
    function removeChainlitWatermark() {
        // CSS pour masquer le watermark
        const style = document.createElement('style');
        style.id = 'sunu-stat-custom-style';
        style.textContent = `
            /* Masquer tous les watermarks Chainlit */
            a[href*="chainlit.io"],
            a[href*="github.com/Chainlit"],
            a[href*="chainlit"],
            [class*="watermark"],
            [data-testid*="watermark"],
            footer,
            [role="contentinfo"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                height: 0 !important;
                overflow: hidden !important;
            }
            
            /* Masquer les conteneurs de watermark avec sélecteurs supportés */
            div[class*="watermark"],
            span[class*="watermark"],
            p[class*="watermark"],
            a[href*="chainlit"] {
                display: none !important;
            }
            
            /* Masquer par attributs data */
            [data-cy*="watermark"],
            [data-testid*="watermark"],
            [id*="watermark"] {
                display: none !important;
            }
            
            /* Style personnalisé pour l'interface Sunu Stat */
            .MuiAppBar-root {
                background: linear-gradient(135deg, #162CF8 0%, #F63D15 100%) !important;
            }
            
            /* Améliorer la visibilité du texte */
            .MuiAppBar-root * {
                color: white !important;
            }
        `;
        
        // Ajouter le style s'il n'existe pas déjà
        if (!document.getElementById('sunu-stat-custom-style')) {
            document.head.appendChild(style);
        }

        // Recherche et suppression manuelle des éléments watermark (méthode robuste)
        try {
            const allElements = document.querySelectorAll('*');
            allElements.forEach(element => {
                const text = element.textContent || element.innerText || '';
                const isWatermark = text.includes('Built with') || 
                                  text.includes('Chainlit') ||
                                  text.includes('Créé avec') ||
                                  (element.href && element.href.includes('chainlit'));
                
                if (isWatermark) {
                    // Algorithme amélioré pour trouver le bon parent à masquer
                    let parentToHide = element;
                    let depth = 0;
                    const maxDepth = 5;
                    
                    while (parentToHide.parentElement && 
                           depth < maxDepth && 
                           parentToHide.parentElement.tagName !== 'BODY' &&
                           parentToHide.parentElement.tagName !== 'MAIN') {
                        
                        const parentText = parentToHide.parentElement.textContent || '';
                        const parentIsSmall = parentText.trim().length < 100; // Augmenté de 50 à 100
                        const parentContainsWatermark = parentText.includes('Built with') || 
                                                      parentText.includes('Chainlit') ||
                                                      parentText.includes('Créé avec');
                        
                        // Conditions plus strictes pour remonter
                        if ((parentIsSmall && parentContainsWatermark) || 
                            (parentText.trim().length < 30)) { // Très petit texte = probablement watermark
                            parentToHide = parentToHide.parentElement;
                            depth++;
                        } else {
                            break;
                        }
                    }
                    
                    // Masquer l'élément trouvé
                    if (parentToHide) {
                        parentToHide.style.setProperty('display', 'none', 'important');
                        parentToHide.style.setProperty('visibility', 'hidden', 'important');
                        parentToHide.style.setProperty('opacity', '0', 'important');
                        parentToHide.style.setProperty('height', '0', 'important');
                        parentToHide.style.setProperty('overflow', 'hidden', 'important');
                        parentToHide.setAttribute('data-sunu-hidden', 'true');
                    }
                }
            });
        } catch (error) {
            console.warn('Erreur lors de la suppression du watermark:', error);
        }
    }

    // Fonction pour traduire les éléments d'interface restants
    function translateInterfaceElements() {
        const translations = {
            'Send message': 'Envoyer le message',
            'Stop task': 'Arrêter',
            'Stop Task': 'Arrêter',
            'Attach files': 'Joindre fichiers',
            'New Chat': 'Nouvelle conversation',
            'Settings': 'Paramètres',
            'Light Theme': 'Thème clair',
            'Dark Theme': 'Thème sombre',
            'Copy to clipboard': 'Copier',
            'Copied!': 'Copié!',
            'Chat': 'Assistant ANSD',
            'Readme': 'À propos',
            'Built with': 'Créé avec',
            'Helpful': 'Utile',
            'Not helpful': 'Pas utile'
        };

        try {
            Object.keys(translations).forEach(englishText => {
                // Recherche plus efficace - seulement les éléments texte
                const textElements = document.querySelectorAll('span, p, button, a, div, h1, h2, h3, h4, h5, h6');
                textElements.forEach(element => {
                    // Ne traduire que les éléments feuilles (sans enfants HTML)
                    if (element.children.length === 0) {
                        const text = element.textContent || element.innerText || '';
                        if (text.trim() === englishText) {
                            element.textContent = translations[englishText];
                        }
                    }
                    
                    // Traduire les attributs importants
                    ['title', 'aria-label', 'alt', 'placeholder'].forEach(attr => {
                        if (element.getAttribute(attr) === englishText) {
                            element.setAttribute(attr, translations[englishText]);
                        }
                    });
                });
            });
        } catch (error) {
            console.warn('Erreur lors de la traduction:', error);
        }
    }

    // Appliquer toutes les personnalisations avec gestion d'erreur
    try {
        updatePlaceholders();
        removeChainlitWatermark();
        translateInterfaceElements();
    } catch (error) {
        console.error('Erreur lors de la personnalisation de l\'interface:', error);
    }
}

// Observer optimisé pour détecter les changements dynamiques
let observerTimeout;
const observer = new MutationObserver((mutations) => {
    // Débouncing pour éviter trop d'exécutions
    clearTimeout(observerTimeout);
    observerTimeout = setTimeout(() => {
        let shouldUpdate = false;
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                // Vérifier si des éléments significatifs ont été ajoutés
                const hasSignificantChanges = Array.from(mutation.addedNodes).some(node => 
                    node.nodeType === 1 && // Element node
                    (node.tagName === 'DIV' || node.tagName === 'SPAN' || node.tagName === 'A' || 
                     node.tagName === 'INPUT' || node.tagName === 'BUTTON')
                );
                if (hasSignificantChanges) {
                    shouldUpdate = true;
                }
            }
        });
        
        if (shouldUpdate) {
            customizeSunuStatInterface();
        }
    }, 150); // Délai de 150ms pour le débouncing
});

// Initialisation robuste quand le DOM est prêt
document.addEventListener('DOMContentLoaded', function() {
    console.log('🇸🇳 Initialisation de Sunu Stat - ANSD');
    
    // Appliquer immédiatement
    customizeSunuStatInterface();
    
    // Démarrer l'observation des changements
    try {
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false, // Désactivé pour les performances
            characterData: false // Désactivé pour les performances
        });
    } catch (error) {
        console.warn('Impossible de démarrer l\'observer:', error);
    }
    
    // Réappliquer après des délais progressifs
    const delays = [500, 1000, 2000, 3000];
    delays.forEach(delay => {
        setTimeout(customizeSunuStatInterface, delay);
    });
});

// Réappliquer quand la fenêtre est complètement chargée
window.addEventListener('load', function() {
    setTimeout(customizeSunuStatInterface, 300);
});

// Réappliquer lors du focus sur la fenêtre (utile pour le développement)
window.addEventListener('focus', function() {
    setTimeout(customizeSunuStatInterface, 100);
});

// Nettoyage lors du déchargement de la page
window.addEventListener('beforeunload', function() {
    if (observer) {
        observer.disconnect();
    }
});