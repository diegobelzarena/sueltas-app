(function(){
    // Configurable double-click threshold (ms). Increase to make double-click detection more forgiving.
    const DOUBLE_CLICK_THRESHOLD = 3000; // 500ms

    // Helper to attach handler to a given plotly graph element
    function attachLegendClickHandler(gd){
        if(!gd || gd.__legendClickHandlerAttached) return;

        let lastClick = { key: null, time: 0 };

        gd.on('plotly_legendclick', function(data){
            try{
                const curveNumber = data.curveNumber;
                const traceName = (data.trace && data.trace.name) ? data.trace.name : String(curveNumber);
                const key = curveNumber + '::' + traceName;
                const now = Date.now();

                        // If same key and within threshold -> double click
                if (lastClick.key === key && (now - lastClick.time) <= DOUBLE_CLICK_THRESHOLD) {
                    // Clear stored click to avoid triple-clicks being considered multi doubles
                    lastClick.key = null; lastClick.time = 0;

                    // Isolation behavior: make only the clicked trace visible, others hidden (legendonly)
                    const visArray = gd.data.map((t, i) => (i === curveNumber ? true : 'legendonly'));
                    Plotly.restyle(gd, {'visible': visArray});

                    // Prevent default behavior for the second click (avoid extra toggles)
                    return false;
                }

                // Not a double click yet — record this click and let Plotly handle single-click toggling.
                lastClick.key = key;
                lastClick.time = now;

                // Clear the buffer after the threshold to avoid stale clicks
                setTimeout(() => {
                    if (lastClick.key === key && (Date.now() - lastClick.time) >= DOUBLE_CLICK_THRESHOLD) {
                        lastClick.key = null; lastClick.time = 0;
                    }
                }, DOUBLE_CLICK_THRESHOLD + 10);

                // Do not prevent Plotly's default single-click legend behavior — let it toggle traces normally.
            }catch(e){
                // Fallback: do nothing and allow default behavior
                console.error('legend-click handler error', e);
                return true;
            }
        });

        // Also attach to double-click event if present to preserve behavior for other listeners
        gd.__legendClickHandlerAttached = true;
    }

    // Attach to existing plots and future plots (plotly_afterplot)
    function init(){
        try{
            document.querySelectorAll('.js-plotly-plot').forEach(attachLegendClickHandler);
            document.addEventListener('plotly_afterplot', function(ev){
                attachLegendClickHandler(ev.target);
            });
        }catch(e){
            console.error('Failed to initialize legend click helpers', e);
        }
    }

    if(document.readyState === 'loading'){
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();