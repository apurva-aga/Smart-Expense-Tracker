// Initialize date picker with today's date
document.addEventListener('DOMContentLoaded', function() {
    // Set today's date as default if it's a date input
    const dateInput = document.getElementById('date');
    if (dateInput) {
        const today = new Date().toISOString().split('T')[0];
        dateInput.value = today;
    }
    
    // Auto-hide flash messages after 5 seconds
    setTimeout(function() {
        const alerts = document.getElementsByClassName('alert');
        for (let i = 0; i < alerts.length; i++) {
            const alert = alerts[i];
            alert.classList.remove('show');
            alert.classList.add('fade');
        }
    }, 5000);
});