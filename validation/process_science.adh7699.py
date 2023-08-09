from Bio import Entrez


def get_snp_info(snp_id):
    Entrez.email = "umarov256@gmail.com"  # Replace with your email
    handle = Entrez.efetch(db="snp", id=snp_id, rettype="gb", retmode="text")
    record = handle.read()
    handle.close()

    return record


def extract_alleles_from_xml(xml_string):
    ref_allele = None
    alt_allele = None

    # Find the position of the <SNP_ID> tag to locate the SNP ID
    snp_id_start = xml_string.find("<SNP_ID>")
    snp_id_end = xml_string.find("</SNP_ID>")
    snp_id = xml_string[snp_id_start + len("<SNP_ID>"): snp_id_end]

    # Find the position of the <DOCSUM> tag to locate the allele information
    docsum_start = xml_string.find("<DOCSUM>")
    docsum_end = xml_string.find("</DOCSUM>")
    docsum = xml_string[docsum_start + len("<DOCSUM>"): docsum_end]

    # Split the DOCSUM text by "|", which separates different allele information
    allele_info_list = docsum.split("|")

    for allele_info in allele_info_list:
        if "SEQ=" in allele_info and "LEN=" in allele_info:
            # Extract the allele information between "SEQ=" and "LEN="
            allele = allele_info.split("SEQ=")[1].split("|")[0]

            # Check if the allele information has more than one nucleotide
            if "/" in allele:
                alleles = allele.split("/")
                ref_allele = alleles[0]
                alt_allele = alleles[1]
            else:
                ref_allele = allele
                alt_allele = allele

            break  # Assuming we found the relevant allele information, we can break the loop

    return snp_id, ref_allele, alt_allele


# Example usage
snp_id = "rs1326279"
record = get_snp_info(snp_id)
snp_id, ref, alt = extract_alleles_from_xml(record)
if ref and alt:
    print(f"SNP ID: {snp_id}")
    print(f"Reference Allele: {ref}")
    print(f"Alternate Allele: {alt}")
else:
    print(f"SNP ID '{snp_id}' not found or information not available.")