$(function() {
    
    // Warnings
	$('dt:has(strong:contains("Warning"))').each(function() {
		var elem = $(this).next('dd');
		var elemText = elem.html();

		$('<div class="alert alert-warning"><strong>Warning</strong> '+elemText+'</div>').insertBefore(this);
		elem.remove();
		$(this).remove();
	});

	// Notes
	$('dt:has(strong:contains("Note"))').each(function() {
		var elem = $(this).next('dd');
		var elemText = elem.html();

		$('<div class="alert alert-info"><strong>Info</strong> '+elemText+'</div>').insertBefore(this);
		elem.remove();
		$(this).remove();
	});

	// See
	$('dt:has(strong:contains("See"))').each(function() {
		var elem = $(this).next('dd');
		var elemText = elem.html();

		var calls = elemText.match(/([a-zA-Z_,\s]+)/)[1];
		var funcs = calls.split(',');
		
		elem.html('');
		for (var i = 0; i < funcs.length; ++i) {
			var func = funcs[i].match(/\s*(MPI_[_a-zA-Z]+)\s*/)[1];
			
			var mpiURL = 'https://www.open-mpi.org/doc/v1.10/man3/'+func+'.3.php';
			elem.append((i == 0 ? '' : ', ')+'<a href="'+mpiURL+'">'+func+'</a>');
		}
	});

});