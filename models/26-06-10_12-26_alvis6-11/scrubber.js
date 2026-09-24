(function() {
    function init() {
        document.querySelectorAll('.scrubber-control[data-audio-steps]').forEach(function(el) {
            // data-initialized guard prevents double-binding when init() is called
            // on dynamically inserted content after DOMContentLoaded
            if (el.dataset.initialized) return;
            el.dataset.initialized = '1';
            var steps = JSON.parse(el.dataset.audioSteps);
            var pathTemplate = el.dataset.audioPath;
            var slider = el.querySelector('.step-slider');
            var display = el.querySelector('.step-display');
            var player = el.querySelector('audio');
            slider.addEventListener('input', function(e) {
                var step = steps[parseInt(e.target.value)];
                display.textContent = step;
                var wasPlaying = !player.paused;
                player.src = pathTemplate.replace('{step}', step);
                player.load();
                if (wasPlaying) player.play();
            });
        });
        document.querySelectorAll('[data-image-steps]').forEach(function(el) {
            if (el.dataset.initialized) return;
            el.dataset.initialized = '1';
            var steps = JSON.parse(el.dataset.imageSteps);
            var slider = el.querySelector('.step-slider');
            var display = el.querySelector('.step-display');
            var img = el.querySelector('img');
            slider.addEventListener('input', function(e) {
                var s = steps[parseInt(e.target.value)];
                display.textContent = s.step;
                img.src = s.src;
            });
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
