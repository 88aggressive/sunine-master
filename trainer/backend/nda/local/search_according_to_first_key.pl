#!/usr/bin/perl

(@ARGV>=2)||die "using $0 [-v] [-c] <searched file>, <key list>\n -v: reverse; -c: clean";

$flag="";
$clean="";

if ($ARGV[0]=~/^-/) {
  if ($ARGV[0] eq "-v"){
    $flag="v";
    shift(@ARGV);
  }elsif($ARGV[0] eq "-c"){
    $clean="T";
    shift(@ARGV);
  }else{

     die "using $0 [-v] [-c] <searched file>, <key list>\n";

   }
  
}

$kfile=$ARGV[1];
$df=$ARGV[0];

open(F1, $kfile)||die "open key file $kfile failed\n";

%box=();

while (<F1>){


chomp;
next if ($_=~/^\s*$/);

$_=~s/^\s+//g;

@o=split(/\s+/, $_);
$k=clean_key($o[0]);

$box{$k}=1;

}

close(F1);

open (F1,$df)||die "open data file $df failed\n";

while (<F1>){

$ll=$_;

chomp($ll);
next if ($ll=~/^\s*$/);

$ll=~s/^\s+//g;
$ll=~s/\s+$//g;

($k)=split(/\s+/,$ll);
$k=clean_key($k);
#print "$_: ",$box{$k},"\n";

if (defined($box{$k})){
  if ($flag ne "v"){
     print $_;
  }
 }else{
    if ($flag eq "v"){
       print $_;

    }

 }

}#all that


sub clean_key
{

  my $key=shift;

  return $key if ($clean eq "") ;
  
  $key=~s/[^a-zA-Z0-9]//g;
  return $key;

}
