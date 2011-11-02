use warnings;

my($file1, $file2, $line1, $line2, $count);

$file1 = shift or die $!;
$file2 = shift or die $!;

open F1, $file1;
open F2, $file2;

while (($line1 = <F1>) && ($line2 = <F2>)) {
    @line1 = split / /, $line1;
    @line2 = split / /, $line2;
    if ($line1[0] ne $line2[0]) {
        $count++;
    }
}

print $count, "\n";

close F1;
close F2;
